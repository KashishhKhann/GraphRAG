"""
Process a batch of notes -> sections -> smart chunks -> store in MongoDB + Neo4j.

Pipeline order after running this:
1. python add_embeddings.py
2. python build_faiss_index.py
3. python kg_extraction.py   (optional but recommended)
4. python kg_rag_query.py    (interactive querying)
"""

import re
from datetime import datetime
from typing import Dict, List

from pymongo import MongoClient
from neo4j import GraphDatabase
from tqdm import tqdm

from config import (
    MONGO_URI, DB_NAME, NOTES_COLLECTION, CHUNKS_COLLECTION, LOG_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FIELD_TEXT, FIELD_CHUNK_ID, FIELD_NOTE_ID,
    FIELD_SUBJECT_ID, FIELD_HADM_ID, FIELD_SECTION,
    validate_neo4j_config
)


# ============================================================
# MAIN PROCESSOR CLASS
# ============================================================
class BatchProcessor:
    def __init__(
        self,
        mongo_uri: str = MONGO_URI,
        db_name: str = DB_NAME,
        notes_collection: str = NOTES_COLLECTION,
        chunks_collection: str = CHUNKS_COLLECTION,
    ):
        # Validate Neo4j configuration before proceeding
        validate_neo4j_config()

        # ----------------- MongoDB -----------------
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.raw_notes = self.db[notes_collection]
        self.chunks_col = self.db[chunks_collection]
        self.log_col = self.db[LOG_COLLECTION]

        # ----------------- Neo4j -------------------
        self.neo4j = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        print("\n================= Batch Pipeline Ready =================")
        print(f"MongoDB       -> {mongo_uri}")
        print(f"Database      -> {db_name}")
        print(f"Source Notes  -> {notes_collection}")
        print(f"Output Chunks -> {chunks_collection}")
        print(f"Log Collection-> {LOG_COLLECTION}")
        print(f"Neo4j         -> {NEO4J_URI}")
        print("========================================================\n")

    # ============================================================
    # 🔹 Batch Runner
    # ============================================================
    def process_batch(self, batch_size: int = 10):
        """Process multiple notes into smart chunks."""

        total = self.raw_notes.count_documents({})
        print(f"📊 Total notes available: {total}")

        processed_ids = {
            log["note_id"] for log in self.log_col.find({"status": "completed"})
        }
        print(f"🗂 Already processed: {len(processed_ids)}\n")

        cursor = self.raw_notes.find(
            {},
            {
                "_id": 1,
                FIELD_TEXT: 1,
                FIELD_SUBJECT_ID: 1,
                FIELD_HADM_ID: 1,
            },
        )

        notes_to_process = []
        for doc in cursor:
            note_id = str(doc["_id"])
            if note_id not in processed_ids:
                notes_to_process.append(doc)
                if len(notes_to_process) >= batch_size:
                    break

        if not notes_to_process:
            print("🎉 All notes already processed.")
            return

        print(f"🚀 Starting batch of {len(notes_to_process)} notes...\n")

        stats = {"success": 0, "failed": 0, "chunks": 0}

        for note in tqdm(notes_to_process, desc="🔨 Processing"):
            note_id = str(note["_id"])
            try:
                chunks = self._process_single(note)
                self.log_col.insert_one(
                    {
                        "note_id": note_id,
                        "status": "completed",
                        "chunks_created": len(chunks),
                        "processed_at": datetime.utcnow(),
                    }
                )
                stats["success"] += 1
                stats["chunks"] += len(chunks)

            except Exception as e:
                self.log_col.insert_one(
                    {
                        "note_id": note_id,
                        "status": "failed",
                        "error": str(e),
                        "processed_at": datetime.utcnow(),
                    }
                )
                stats["failed"] += 1

        print("\n================ BATCH COMPLETE ================")
        print(f"  ✔ Success     → {stats['success']}")
        print(f"  ❌ Failed      → {stats['failed']}")
        print(f"  🧩 Chunks Made → {stats['chunks']}")
        print("================================================\n")

        if stats["success"] > 0:
            print("➡ Next steps:")
            print("1. python add_embeddings.py")
            print("2. python build_faiss_index.py")
            print("3. python kg_extraction.py (optional)")
            print("4. python kg_rag_query.py\n")

        return stats

    # ============================================================
    # 🔹 Note → Smart Chunks
    # ============================================================
    def _process_single(self, note: Dict) -> List[Dict]:
        """Parse sections -> smart-chunk within each section -> store -> Neo4j."""
        note_id = str(note["_id"])
        text = note.get(FIELD_TEXT) or ""

        if not text.strip():
            text = str(note)

        # Safety: skip if we've already created chunks for this note
        existing = self.chunks_col.count_documents({FIELD_NOTE_ID: note_id})
        if existing > 0:
            print(f"SKIPPED: chunks for note_id={note_id} already exist ({existing})")
            return []

        sections = self._parse_sections(text)

        all_chunks: List[Dict] = []
        global_idx = 0

        for section_name, section_text in sections.items():
            if not section_text or len(section_text.strip()) < 30:
                continue

            smart_chunks = self._smart_chunk_section(section_text)

            for _local_idx, chunk_text in enumerate(smart_chunks):
                if len(chunk_text.strip()) < 30:
                    continue

                metadata = self._extract_metadata(chunk_text, section_name)

                chunk_doc = {
                    # Core traceability
                    FIELD_CHUNK_ID: f"{note_id}_{section_name}_{global_idx}",
                    FIELD_NOTE_ID: note_id,
                    FIELD_SUBJECT_ID: note.get(FIELD_SUBJECT_ID),
                    FIELD_HADM_ID: note.get(FIELD_HADM_ID),

                    # Content
                    FIELD_SECTION: section_name,
                    "text": chunk_text,
                    "full_text": (
                        f"[Section: {section_name} | Temporal: {metadata['temporal']}]\n"
                        f"---\n{chunk_text}"
                    ),
                    "metadata": metadata,
                    "chunk_index": global_idx,
                    "created_at": datetime.utcnow(),
                }

                all_chunks.append(chunk_doc)
                global_idx += 1

        if all_chunks:
            self.chunks_col.insert_many(all_chunks)
            self._push_to_neo4j(note_id, all_chunks)

        return all_chunks

    # ============================================================
    # 🔹 Section Parsing (regex / marker-based)
    # ============================================================
    def _parse_sections(self, text: str) -> Dict[str, str]:
        """
        Parse note into sections using marker phrases, case-insensitive.
        If no markers are found, entire note becomes one 'body' section.
        """
        sections: Dict[str, str] = {}
        current = "header"
        buffer: List[str] = []

        markers = {
            "chief complaint": "chief_complaint",
            "history of present illness": "hpi",
            "past medical history": "pmh",
            "physical exam": "physical_exam",
            "hospital course": "hospital_course",
            "brief hospital course": "hospital_course",
            "discharge medications": "meds_discharge",
        }

        for line in text.split("\n"):
            low = line.lower()

            matched_key = None
            for phrase, sec_key in markers.items():
                if phrase in low:
                    matched_key = sec_key
                    break

            if matched_key:
                if buffer:
                    sections[current] = "\n".join(buffer).strip()
                current = matched_key
                buffer = []
            else:
                buffer.append(line)

        if buffer:
            sections[current] = "\n".join(buffer).strip()

        if len(sections) == 1 and "header" in sections:
            sections = {"body": sections["header"]}

        return sections

    # ============================================================
    # 🔹 Smart Hybrid Chunking
    # ============================================================
    def _smart_chunk_section(
        self,
        text: str,
        target_len: int = 900,
        max_len: int = 1300,
        min_len: int = 300,
    ) -> List[str]:

        text = text.strip()
        if len(text) <= max_len:
            return [text]

        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [text]

        chunks: List[str] = []
        current_buf: List[str] = []
        current_len = 0

        def flush_current():
            nonlocal current_buf, current_len
            if current_buf:
                chunk_text = "\n\n".join(current_buf).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_buf = []
                current_len = 0

        for para in raw_paragraphs:
            plen = len(para)

            if plen > max_len:
                flush_current()
                chunks.extend(self._split_by_sentences(para, target_len, max_len, min_len))
                continue

            if current_len + plen <= target_len:
                current_buf.append(para)
                current_len += plen + 2
            else:
                flush_current()
                current_buf.append(para)
                current_len = plen

        flush_current()

        final_chunks: List[str] = []
        for ch in chunks:
            if len(ch) > max_len:
                final_chunks.extend(
                    self._split_by_sentences(ch, target_len, max_len, min_len)
                )
            else:
                final_chunks.append(ch)

        return final_chunks

    def _split_by_sentences(
        self,
        text: str,
        target_len: int,
        max_len: int,
        min_len: int,
    ) -> List[str]:

        sentences = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", text) if s.strip()]
        if not sentences:
            return [text]

        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        def flush_buf():
            nonlocal buf, buf_len
            if buf:
                ch = " ".join(buf).strip()
                if ch:
                    chunks.append(ch)
                buf = []
                buf_len = 0

        for sent in sentences:
            slen = len(sent)

            if slen > max_len:
                flush_buf()
                chunks.append(sent)
                continue

            if buf_len + slen <= target_len or buf_len < min_len:
                buf.append(sent)
                buf_len += slen + 1
            else:
                flush_buf()
                buf.append(sent)
                buf_len = slen + 1

        flush_buf()
        return chunks

    # ============================================================
    # 🔹 Metadata extraction
    # ============================================================
    def _extract_metadata(self, text: str, section: str) -> Dict:
        lo = text.lower()
        if any(w in lo for w in ["admission", "admitted", "presented"]):
            temporal = "admission"
        elif "discharge" in lo or "discharged" in lo:
            temporal = "discharge"
        else:
            temporal = "during_stay"

        return {
            "section": section,
            "temporal": temporal,
            "has_medications": bool(re.search(r"\b\d+\s*mg\b", text)),
            "has_labs": bool(re.search(r"\b(Na|K|Cr)\s*[:=]?\s*\d", text)),
        }

    # ============================================================
    # 🔹 Push simple Note–Chunk structure to Neo4j
    # ============================================================
    def _push_to_neo4j(self, note_id: str, chunks: List[Dict]):
        with self.neo4j.session() as session:
            session.run(
                """
                MERGE (n:Note {note_id: $note_id})
                ON CREATE SET n.created_at = datetime()
                """,
                note_id=note_id,
            )

            for c in chunks:
                session.run(
                    """
                    MATCH (n:Note {note_id: $note_id})
                    MERGE (ch:Chunk {chunk_id: $chunk_id})
                    SET ch.section = $section
                    MERGE (n)-[:HAS_CHUNK]->(ch)
                    """,
                    note_id=note_id,
                    chunk_id=c["chunk_id"],
                    section=c["section"],
                )

    # ============================================================
    # 🔹 Cleanup
    # ============================================================
    def cleanup(self):
        self.client.close()
        self.neo4j.close()


# --------------------------------------------------------------
def main():
    try:
        bs = input("\nBatch size? (default 10): ").strip()
        batch_size = int(bs) if bs.isdigit() else 10

        processor = BatchProcessor()
        processor.process_batch(batch_size)

    except KeyboardInterrupt:
        print("\nCancelled by user")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            processor.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
