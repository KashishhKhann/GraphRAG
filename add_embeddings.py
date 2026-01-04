"""
Add BioClinicalBERT embeddings to all chunks in MIMIC.processed_chunks.

- Uses local MongoDB only
- Skips chunks that already have an embedding (idempotent)
"""

from pymongo import MongoClient
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import (
    MONGO_URI, DB_NAME, CHUNKS_COLLECTION, BERT_MODEL,
    FIELD_EMBEDDING, FIELD_FULL_TEXT, FIELD_TEXT
)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[CHUNKS_COLLECTION]

print("\n============================================================")
print("Adding BioClinicalBERT embeddings to processed_chunks")
print("============================================================")
print(f"MongoDB: {MONGO_URI}")
print(f"Database: {DB_NAME}")
print(f"Collection: {CHUNKS_COLLECTION}")
print(f"Model: {BERT_MODEL}")

embedder = SentenceTransformer(BERT_MODEL)
dim = embedder.get_sentence_embedding_dimension()
print(f"Embedding dimension: {dim}")

# Only chunks without "embedding" field
cursor = col.find(
    {FIELD_EMBEDDING: {"$exists": False}},
    {"_id": 1, FIELD_FULL_TEXT: 1, FIELD_TEXT: 1},
)
missing_count = col.count_documents({FIELD_EMBEDDING: {"$exists": False}})
print(f"Chunks missing embeddings: {missing_count}")

if missing_count == 0:
    print("No work to do. All chunks already have embeddings.")
else:
    for doc in tqdm(cursor, total=missing_count, desc="Embedding chunks"):
        text = doc.get(FIELD_FULL_TEXT) or doc.get(FIELD_TEXT) or ""
        if not text.strip():
            continue

        vec = embedder.encode(text)
        vec = vec.tolist()

        col.update_one({"_id": doc["_id"]}, {"$set": {FIELD_EMBEDDING: vec}})

print("\n============================================================")
print("Finished embedding processed_chunks")
print("============================================================\n")

client.close()
