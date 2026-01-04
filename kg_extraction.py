"""
KG Extraction from processed_chunks -> Neo4j

For each chunk in MIMIC.processed_chunks:
- run spaCy NER
- create Concept nodes in Neo4j
- connect Chunk -> Concept with MENTIONS_CONCEPT

This is a lightweight, robust KG just for signal & reasoning.
"""

from typing import List

from pymongo import MongoClient
from neo4j import GraphDatabase
from tqdm import tqdm

import spacy

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    FIELD_CHUNK_ID, FIELD_TEXT, FIELD_KG_STATUS,
    validate_neo4j_config
)

# ------------------------------------------------------------------
# spaCy model loading (SciSpaCy if available, else en_core_web_sm)
# ------------------------------------------------------------------
def load_nlp():
    try:
        # try a SciSpaCy model first (if installed)
        return spacy.load("en_core_sci_sm")
    except Exception:
        try:
            return spacy.load("en_core_web_sm")
        except Exception as e:
            print("❌ Could not load any spaCy model (en_core_sci_sm or en_core_web_sm).")
            raise e

nlp = load_nlp()

# ------------------------------------------------------------------
# DB clients
# ------------------------------------------------------------------
validate_neo4j_config()

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
chunks_col = db[CHUNKS_COLLECTION]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# We'll track which chunks already processed via kg_status flag
def get_chunks_to_process(limit: int | None = None):
    query = {FIELD_KG_STATUS: {"$ne": "done"}}
    cursor = chunks_col.find(query, {FIELD_CHUNK_ID: 1, FIELD_TEXT: 1})
    if limit:
        cursor = cursor.limit(limit)
    return cursor

def process_chunk(tx, chunk_id: str, concepts: List[str]):
    # Ensure Chunk node exists
    tx.run(
        """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        """,
        chunk_id=chunk_id,
    )

    for name in concepts:
        if not name:
            continue
        tx.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (e:Concept {name: $name})
            MERGE (c)-[:MENTIONS_CONCEPT {source:'spacy'}]->(e)
            """,
            chunk_id=chunk_id,
            name=name,
        )

def extract_concepts(text: str) -> List[str]:
    doc = nlp(text)
    ents = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
    # de-duplicate while preserving order
    seen = set()
    out = []
    for e in ents:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

def main():
    print("\n============================================================")
    print("Running KG Extraction over processed_chunks -> Neo4j")
    print("============================================================")
    print(f"MongoDB: {MONGO_URI}")
    print(f"Database: {DB_NAME}")
    print(f"Collection: {CHUNKS_COLLECTION}")
    print(f"Neo4j: {NEO4J_URI}")
    print("============================================================\n")

    total = chunks_col.count_documents({FIELD_KG_STATUS: {"$ne": "done"}})
    print(f"Chunks to process ({FIELD_KG_STATUS} != 'done'): {total}")

    if total == 0:
        print("Nothing to do. All chunks already processed.")
        return

    count_done = 0
    count_failed = 0

    chunks = get_chunks_to_process()
    for doc in tqdm(chunks, total=total, desc="KG extracting"):
        cid = doc[FIELD_CHUNK_ID]
        text = doc.get(FIELD_TEXT) or ""
        if not text.strip():
            chunks_col.update_one(
                {"_id": doc["_id"]}, {"$set": {FIELD_KG_STATUS: "skipped_empty"}}
            )
            continue

        try:
            concepts = extract_concepts(text)
            with driver.session() as session:
                session.execute_write(process_chunk, cid, concepts)

            chunks_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {FIELD_KG_STATUS: "done", "kg_concepts_count": len(concepts)}},
            )
            count_done += 1

        except Exception as e:
            print(f"\nError on chunk {cid}: {e}")
            chunks_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {FIELD_KG_STATUS: "error", "kg_error": str(e)}},
            )
            count_failed += 1

    print("\n================ KG EXTRACTION COMPLETE ================")
    print(f"  Processed chunks : {count_done}")
    print(f"  Failed chunks    : {count_failed}")
    print("========================================================\n")

    mongo_client.close()
    driver.close()


if __name__ == "__main__":
    main()
