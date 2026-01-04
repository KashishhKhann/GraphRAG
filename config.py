"""
Centralized configuration for the Graph RAG pipeline.

All scripts should import from this module to ensure consistency.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# MongoDB Configuration
# ============================================================
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB", "MIMIC")

# Collections
NOTES_COLLECTION = os.getenv("MONGODB_NOTES_COLLECTION", "filtered_notes")
CHUNKS_COLLECTION = os.getenv("MONGODB_CHUNKS_COLLECTION", "processed_chunks")
LOG_COLLECTION = "processing_log"

# ============================================================
# Neo4j Configuration
# ============================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "MIMIC")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "MIMIC@123")

# ============================================================
# FAISS Configuration
# ============================================================
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_biobert.index")
FAISS_MAP_PATH = os.getenv("FAISS_MAP_PATH", "data/faiss_biobert_mapping.json")

# ============================================================
# Embedding Model Configuration
# ============================================================
# Options (safetensors-only models that work with any PyTorch version):
#   - "sentence-transformers/all-mpnet-base-v2" (general, high quality, 768-dim)
#   - "sentence-transformers/all-MiniLM-L6-v2" (fast, smaller, 384-dim)
#   - "sentence-transformers/multi-qa-mpnet-base-dot-v1" (good for Q&A)
# Medical models (require PyTorch >= 2.6):
#   - "emilyalsentzer/Bio_ClinicalBERT"
#   - "pritamdeka/S-PubMedBert-MS-MARCO"
BERT_MODEL = os.getenv("BIOCLINICALBERT_MODEL", "sentence-transformers/all-mpnet-base-v2")

# ============================================================
# LLM Configuration (Ollama)
# ============================================================
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "medllama2:latest")

# ============================================================
# Document Schema Field Names (for consistency)
# ============================================================
FIELD_TEXT = "text"
FIELD_FULL_TEXT = "full_text"
FIELD_CHUNK_ID = "chunk_id"
FIELD_NOTE_ID = "note_id"
FIELD_SUBJECT_ID = "subject_id"
FIELD_HADM_ID = "hadm_id"
FIELD_SECTION = "section"
FIELD_EMBEDDING = "embedding"
FIELD_KG_STATUS = "kg_status"

# ============================================================
# Load optional data_config.json for backward compatibility
# ============================================================
DATA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data_config.json")

def load_data_config():
    """Load data_config.json if it exists, for backward compatibility."""
    if os.path.exists(DATA_CONFIG_PATH):
        with open(DATA_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

# Override from data_config.json if present
_data_config = load_data_config()
if _data_config.get("database"):
    DB_NAME = _data_config["database"]
if _data_config.get("collection"):
    NOTES_COLLECTION = _data_config["collection"]
if _data_config.get("text_field"):
    FIELD_TEXT = _data_config["text_field"]

# ============================================================
# Validation helpers
# ============================================================
def validate_neo4j_config():
    """Check that Neo4j configuration is present."""
    if not NEO4J_URI:
        raise ValueError("NEO4J_URI environment variable is not set")
    if not NEO4J_USER:
        raise ValueError("NEO4J_USER environment variable is not set")
    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD environment variable is not set")

def print_config():
    """Print current configuration (for debugging)."""
    print("\n============================================================")
    print("GRAPH RAG CONFIGURATION")
    print("============================================================")
    print(f"MongoDB URI:      {MONGO_URI}")
    print(f"Database:         {DB_NAME}")
    print(f"Notes Collection: {NOTES_COLLECTION}")
    print(f"Chunks Collection:{CHUNKS_COLLECTION}")
    print(f"Neo4j URI:        {NEO4J_URI}")
    print(f"Neo4j User:       {NEO4J_USER}")
    print(f"FAISS Index:      {FAISS_INDEX_PATH}")
    print(f"FAISS Mapping:    {FAISS_MAP_PATH}")
    print(f"Embedding Model:  {BERT_MODEL}")
    print(f"LLM URL:          {OLLAMA_URL}")
    print(f"LLM Model:        {OLLAMA_MODEL}")
    print("============================================================\n")


if __name__ == "__main__":
    print_config()
