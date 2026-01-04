"""
KG-aware RAG query script (Hybrid: Short Answer + Reasoning + Evidence)

Components:
- Embeddings:    BioClinicalBERT (sentence-transformers)
- Vector store:  FAISS (data/faiss_biobert.index + mapping)
- Store:         MongoDB (MIMIC.processed_chunks)
- KG:            Neo4j (Chunk, Concept via MENTIONS_CONCEPT)
- LLM:           Ollama (model set via OLLAMA_MODEL)

Features:
- Filters by patient (subject_id) and admission (hadm_id)
- Uses hybrid ranking (FAISS similarity + KG concept density)
- Answer structure:
    1) Short direct answer
    2) Reasoning chain
    3) Evidence summary (chunks + patient/admission)
"""

import os
import json
import textwrap
from typing import List, Dict, Any, Tuple

import numpy as np
from pymongo import MongoClient
from neo4j import GraphDatabase
import faiss
import requests
import spacy
from sentence_transformers import SentenceTransformer

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    FAISS_INDEX_PATH, FAISS_MAP_PATH,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_URL, OLLAMA_MODEL, BERT_MODEL,
    FIELD_CHUNK_ID, FIELD_SUBJECT_ID, FIELD_HADM_ID,
    FIELD_SECTION, FIELD_TEXT,
    validate_neo4j_config
)


# ---------------------------------------------------------
# Init helpers
# ---------------------------------------------------------
def init_mongo():
    client = MongoClient(MONGO_URI)
    return client, client[DB_NAME][CHUNKS_COLLECTION]


def init_faiss() -> Tuple[faiss.Index, Dict[int, str]]:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS index at {FAISS_INDEX_PATH}")
    if not os.path.exists(FAISS_MAP_PATH):
        raise FileNotFoundError(f"Missing FAISS mapping at {FAISS_MAP_PATH}")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "r") as f:
        mapping_raw = json.load(f)
    mapping = {int(k): v for k, v in mapping_raw.items()}

    print(f"FAISS Loaded | dim={index.d} | entries={len(mapping)}")
    return index, mapping


def init_embedder():
    print(f"\nLoading embedding model -> {BERT_MODEL}")
    model = SentenceTransformer(BERT_MODEL)
    return model


def init_neo4j():
    validate_neo4j_config()
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ---------------------------------------------------------
# spaCy model loading
# ---------------------------------------------------------
def load_nlp():
    try:
        return spacy.load("en_core_sci_sm")
    except Exception:
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
            if "ner" not in nlp.pipe_names:
                nlp.add_pipe("ner")
            return nlp


nlp = load_nlp()


# ---------------------------------------------------------
# Embedding + FAISS retrieval
# ---------------------------------------------------------
def embed_query(model, text: str) -> np.ndarray:
    vec = model.encode(text)
    q = np.asarray(vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    return q


def faiss_candidates(
    index: faiss.Index,
    mapping: Dict[int, str],
    q_vec: np.ndarray,
    oversample: int = 60,
) -> List[Dict[str, Any]]:
    D, I = index.search(q_vec, oversample)
    hits: List[Dict[str, Any]] = []
    print("📎 FAISS neighbours (index → score):")
    for idx, dist in zip(I[0], D[0]):
        if idx == -1:
            continue
        cid = mapping.get(int(idx))
        if not cid:
            continue
        print(f"   • {idx} → chunk_id={cid}, score={dist:.4f}")
        hits.append({"faiss_id": int(idx), "chunk_id": cid, "faiss_score": float(dist)})
    return hits


def attach_docs_and_filter(
    hits: List[Dict[str, Any]],
    coll,
    top_k: int,
    subject_id: int | None,
    hadm_id: int | None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        cid = h["chunk_id"]
        doc = coll.find_one({FIELD_CHUNK_ID: cid})
        if not doc:
            continue

        if subject_id is not None and doc.get(FIELD_SUBJECT_ID) != subject_id:
            continue
        if hadm_id is not None and doc.get(FIELD_HADM_ID) != hadm_id:
            continue

        out.append(
            {
                "chunk_id": cid,
                "faiss_score": h["faiss_score"],
                "doc": doc,
            }
        )

        if len(out) >= top_k * 3:
            break

    return out


# ---------------------------------------------------------
# KG helpers
# ---------------------------------------------------------
def get_concepts_for_chunk(driver, chunk_id: str) -> List[str]:
    """
    Read Concept nodes connected via MENTIONS_CONCEPT.
    """
    q = """
    MATCH (c:Chunk {chunk_id:$cid})-[:MENTIONS_CONCEPT]->(e:Concept)
    RETURN DISTINCT e.name AS name
    """
    names: List[str] = []
    with driver.session() as s:
        for r in s.run(q, cid=chunk_id):
            nm = r.get("name")
            if nm:
                names.append(nm)
    return names


def extract_query_concepts(text: str) -> List[str]:
    doc = nlp(text)
    ents = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
    seen = set()
    out = []
    for e in ents:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def compute_kg_overlap(query_concepts: set[str], chunk_concepts: List[str]) -> float:
    """
    Overlap score via Jaccard similarity between query and chunk concepts.
    """
    if not query_concepts or not chunk_concepts:
        return 0.0
    chunk_set = set(chunk_concepts)
    union = query_concepts | chunk_set
    if not union:
        return 0.0
    return float(len(query_concepts & chunk_set) / len(union))


def hybrid_rank(
    candidates: List[Dict[str, Any]],
    driver,
    query_concepts: set[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    enriched: List[Dict[str, Any]] = []

    for c in candidates:
        cid = c["chunk_id"]
        concepts = get_concepts_for_chunk(driver, cid)
        kg_score = compute_kg_overlap(query_concepts, concepts)
        enriched.append(
            {
                **c,
                "concepts": concepts,
                "kg_score": kg_score,
            }
        )

    faiss_vals = [e["faiss_score"] for e in enriched]
    kg_vals = [e["kg_score"] for e in enriched]

    def minmax(values):
        if not values:
            return []
        mn, mx = min(values), max(values)
        if mx == mn:
            return [1.0] * len(values)
        return [(v - mn) / (mx - mn + 1e-9) for v in values]

    faiss_norm = minmax(faiss_vals)
    kg_norm = minmax(kg_vals)

    alpha = 0.7
    beta = 0.3
    for e, fn, kn in zip(enriched, faiss_norm, kg_norm):
        e["hybrid_score"] = alpha * fn + beta * kn

    enriched.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return enriched[:top_k]


# ---------------------------------------------------------
# Context building
# ---------------------------------------------------------
def build_context_block(rank_idx: int, cand: Dict[str, Any]) -> str:
    doc = cand["doc"]
    concepts = cand.get("concepts", [])

    chunk_id = doc.get("chunk_id")
    section = doc.get("section", "unknown")
    meta = doc.get("metadata", {})
    temporal = meta.get("temporal", "unknown")

    subject_id = doc.get("subject_id")
    hadm_id = doc.get("hadm_id")

    text = doc.get("text", "") or ""
    snippet = textwrap.shorten(text.replace("\n", " "), width=420, placeholder=" ...")

    lines: List[str] = []
    lines.append(
        f"### [{rank_idx}] chunk_id={chunk_id} "
        f"| patient={subject_id} | hadm={hadm_id} "
        f"| section={section} | temporal={temporal}"
    )
    lines.append(f"TEXT SNIPPET:\n{snippet}\n")

    if concepts:
        lines.append("CONCEPTS: " + "; ".join(sorted(set(concepts))))

    return "\n".join(lines)


def build_full_context(cands: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(cands, start=1):
        blocks.append(build_context_block(i, c))
    return "\n\n".join(blocks)


# ---------------------------------------------------------
# LLM call
# ---------------------------------------------------------
def call_llm(question: str, context: str, subject_id=None, hadm_id=None) -> str:
    system = """
You are a clinical decision-support assistant using a RAG + KG pipeline.

You are given:
- Retrieved note chunks with chunk_id, patient_id (subject_id), hadm_id, section, temporal.
- Concept-level KG facts from the notes.

Your job:
1. Provide a SHORT DIRECT ANSWER (3–6 sentences) to the question.
2. Then provide a REASONING SECTION as bullet points:
   - how the evidence supports the answer
   - how treatments relate to diagnoses/symptoms
   - any major uncertainties
3. Then provide an EVIDENCE SUMMARY:
   - list the most relevant chunk_ids and sections you used
   - mention patient_id and hadm_id for each
   - use the format [chunk:chunk_id].

Do NOT hallucinate medications, diagnoses, or dates that are not supported by the context.
If evidence is weak or incomplete, say that explicitly.
"""

    prompt = f"""{system}

------------------------------------------------
CLINICAL QUESTION:
{question}

Patient filter:   {subject_id if subject_id is not None else "all"}
Admission filter: {hadm_id if hadm_id is not None else "all"}
------------------------------------------------

RETRIEVED CONTEXT:
{context}
------------------------------------------------

Now answer in this structure:

1. Short Answer
2. Reasoning (bullet points)
3. Evidence Summary (bullet list with [chunk:chunk_id])

ANSWER:
"""

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
            },
        },
        timeout=180,
    )
    data = resp.json()
    return data.get("response", "").strip()


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def kg_rag_query(
    question: str,
    top_k: int = 5,
    subject_id: int | None = None,
    hadm_id: int | None = None,
):
    print("\n============================================================")
    print("KG-AWARE MEDICAL RAG QUERY (Hybrid KG Mode)")
    print("============================================================")
    print(f"MongoDB:   {MONGO_URI}  DB={DB_NAME}  Coll={CHUNKS_COLLECTION}")
    print(f"Neo4j:     {NEO4J_URI}")
    print(f"FAISS:     {FAISS_INDEX_PATH}")
    print(f"LLM:       {OLLAMA_MODEL}")
    print("============================================================")

    mongo_client, coll = init_mongo()
    index, mapping = init_faiss()
    embedder = init_embedder()
    driver = init_neo4j()

    try:
        print(f"\nQUESTION: {question}")
        print(f"Filter: {FIELD_SUBJECT_ID}={subject_id} | {FIELD_HADM_ID}={hadm_id}\n")

        query_concepts = set(extract_query_concepts(question))
        q_vec = embed_query(embedder, question)
        hits_raw = faiss_candidates(index, mapping, q_vec, oversample=top_k * 20)

        if not hits_raw:
            print("No FAISS hits for this query.")
            return None

        filtered = attach_docs_and_filter(
            hits_raw, coll, top_k=top_k, subject_id=subject_id, hadm_id=hadm_id
        )

        if not filtered:
            print("No chunks matched the patient/admission filters.")
            return None

        ranked = hybrid_rank(filtered, driver, query_concepts, top_k=top_k)
        print(f"\nUsing top {len(ranked)} chunks after hybrid scoring.\n")

        context = build_full_context(ranked)
        answer = call_llm(question, context, subject_id, hadm_id)

        print("\n================= FINAL ANSWER =================\n")
        print(answer)
        print("\n================================================\n")

        return answer

    finally:
        mongo_client.close()
        driver.close()


# ---------------------------------------------------------
# CLI entry
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n============================================================")
    print("KG-AWARE MEDICAL RAG SYSTEM (Medtron + Hybrid KG Mode)")
    print("============================================================")

    try:
        q = input("Enter your medical question:\n> ").strip()
        if not q:
            print("No question entered, exiting.")
            raise SystemExit

        pid_str = input("Filter by patient_id (subject_id) [blank = all]: ").strip()
        hadm_str = input("Filter by hadm_id [blank = all]: ").strip()

        subject_id = int(pid_str) if pid_str.isdigit() else None
        hadm_id = int(hadm_str) if hadm_str.isdigit() else None

        kg_rag_query(q, top_k=5, subject_id=subject_id, hadm_id=hadm_id)

    except KeyboardInterrupt:
        print("\nCancelled by user.")
