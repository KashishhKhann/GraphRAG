"""
FAISS search with optional patient-level filtering.
"""

import os
import faiss
import numpy as np
import json
from pymongo import MongoClient
from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION,
    FAISS_INDEX_PATH, FAISS_MAP_PATH,
    FIELD_CHUNK_ID, FIELD_SUBJECT_ID, FIELD_HADM_ID, FIELD_SECTION
)


def load_faiss_index():
    """Load FAISS index and mapping with proper key type conversion."""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    if not os.path.exists(FAISS_MAP_PATH):
        raise FileNotFoundError(f"FAISS mapping not found at {FAISS_MAP_PATH}")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_MAP_PATH, "r") as f:
        mapping_raw = json.load(f)

    # Convert string keys to int for consistent lookup
    # FAISS returns integer indices, so we need int keys
    mapping = {int(k): v for k, v in mapping_raw.items()}

    return index, mapping


# Initialize on module load
_faiss_index, _id_map = load_faiss_index()

# MongoDB connection
_client = MongoClient(MONGO_URI)
_db = _client[DB_NAME]
_chunks_col = _db[CHUNKS_COLLECTION]


def faiss_search_filtered(query_embedding, top_k=10, patient_id=None, hadm_id=None):
    """
    Search FAISS index with optional patient/admission filtering.

    Args:
        query_embedding: Pre-computed embedding vector (should be normalized)
        top_k: Number of results to return
        patient_id: Optional subject_id filter
        hadm_id: Optional hospital admission ID filter

    Returns:
        List of matching chunks with scores
    """
    query = np.array(query_embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query)

    # Oversample to account for filtering
    D, I = _faiss_index.search(query, top_k * 3)

    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue

        # Use integer key lookup (fixed from str)
        chunk_id = _id_map.get(int(idx))
        if not chunk_id:
            continue

        doc = _chunks_col.find_one({FIELD_CHUNK_ID: chunk_id})
        if not doc:
            continue

        # Apply filters
        if patient_id is not None and doc.get(FIELD_SUBJECT_ID) != patient_id:
            continue
        if hadm_id is not None and doc.get(FIELD_HADM_ID) != hadm_id:
            continue

        results.append({
            FIELD_CHUNK_ID: chunk_id,
            FIELD_SUBJECT_ID: doc.get(FIELD_SUBJECT_ID),
            FIELD_HADM_ID: doc.get(FIELD_HADM_ID),
            FIELD_SECTION: doc.get(FIELD_SECTION),
            "score": float(dist),
        })

        if len(results) >= top_k:
            break

    return results


def close_connections():
    """Clean up MongoDB connection."""
    _client.close()
