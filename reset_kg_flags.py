"""
Reset the KG processing flag on all processed chunks.
This allows kg_extraction.py to run again from scratch.
"""

from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, CHUNKS_COLLECTION, FIELD_KG_STATUS

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll = db[CHUNKS_COLLECTION]

# Remove the kg_status flag from all documents
# This matches the field name used in kg_extraction.py
result = coll.update_many(
    {},
    {"$unset": {FIELD_KG_STATUS: "", "kg_concepts_count": "", "kg_error": ""}}
)

print(f"Cleared KG flags ({FIELD_KG_STATUS}, kg_concepts_count, kg_error) for {result.modified_count} chunks.")

client.close()
