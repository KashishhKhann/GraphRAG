# 🏥 Medical RAG System

### **Graph-Based Retrieval-Augmented Generation for MIMIC-IV Clinical Notes**

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline for
semantic search, structured retrieval, and question-answering over
**MIMIC-IV clinical notes**.\
This system combines **vector search**, **knowledge-graph reasoning**,
and **local LLM inference** to deliver a fully local, privacy-preserving
clinical NLP workflow suitable for research, prototyping, and real-world
clinical decision support experiments.

------------------------------------------------------------------------

## 🚀 Key Features

### **Hybrid Retrieval Architecture**

-   **MongoDB Vector Search** (384-dim embeddings)
-   **Neo4j Knowledge Graph** for clinical entity relationships
-   **Hybrid fusion** (semantic similarity + graph reasoning)

### **Advanced Clinical Text Processing**

-   Context-aware chunking preserving note structure (HPI, PMH, ROS...)
-   Section headers, temporal markers, and narrative context prefixes
-   Entity extraction (diagnosis, medication, symptom, procedure)

### **Local, Privacy-First Inference**

-   Runs fully offline using **Ollama** + **Llama 3.2 3B**
-   Optional larger or biomedical models (Llama 3.1 8B, BioMistral 7B)
-   Zero API usage required

### **Reranking & Hybrid Reasoning**

-   Optional **cross-encoder reranking** (MS-MARCO MiniLM)
-   Graph-enhanced retrieval: diagnoses → symptoms → meds

### **Production-Oriented Design**

-   Validated on real MIMIC-IV notes\
-   Modular ingestion pipeline\
-   Clean project architecture with environment-based configuration

------------------------------------------------------------------------

## 🧱 System Architecture

                         ┌──────────────────────────────┐
                         │    Raw MIMIC Clinical Notes   │
                         └───────────────┬───────────────┘
                                         │
                                 Ingestion Pipeline
                              (section parsing, entities,
                             metadata, embeddings, chunks)
                                         │
                   ┌─────────────────────┼──────────────────────┐
                   │                                             │
       ┌─────────────────────┐                         ┌──────────────────────┐
       │   MongoDB Vector     │                         │     Neo4j Graph      │
       │   Store (Chunks)     │                         │ (Diagnoses, Meds…)   │
       └──────────┬───────────┘                         └──────────┬───────────┘
                  │                                                │
                  └──────────────────────┬─────────────────────────┘
                                         │
                                  Query Engine (RAG)
                       vector search → graph reasoning → reranking
                                         │
                                  Local LLM via Ollama
                                         │
                                   Final Answer + Sources

------------------------------------------------------------------------

## 🛠️ Quick Start

### **Prerequisites**

-   macOS/Linux\
-   Python 3.9+\
-   MongoDB (local or Atlas)\
-   Docker (for Neo4j)\
-   16GB RAM recommended\
-   Ollama installed locally

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
# Clone
git clone https://github.com/yourusername/medical-rag-system.git
cd medical-rag-system

# Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: Install Ollama + LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# Start Neo4j
docker run --name neo4j-medical   -p 7474:7474 -p 7687:7687   -e NEO4J_AUTH=neo4j/medical123   -d neo4j:latest

# Configure environment
cp .env.example .env
```

Edit `.env`:

``` bash
MONGODB_URI=mongodb+srv://your-connection-string
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=medical123
OLLAMA_MODEL=llama3.2:3b
```

------------------------------------------------------------------------

## 🔍 Verifying Setup

``` bash
python scripts/test_connections.py
```

Expected:

    ✅ Connected to MongoDB
    ✅ Connected to Neo4j
    ✅ Ollama running
    🎉 All systems operational!

------------------------------------------------------------------------

## 📊 Processing Data

### **Single Note**

``` bash
python scripts/process_first_note.py
```

### **Batch Processing**

``` bash
python scripts/process_batch.py
```

### **Generate Embeddings**

``` bash
python scripts/add_embeddings.py
```

------------------------------------------------------------------------

## ❓ Querying the System

### Interactive CLI

``` bash
python scripts/first_query.py
```

### Example Queries

-   "What was the primary diagnosis?"\
-   "Why was the patient readmitted?"\
-   "What medications were prescribed at discharge?"

### Programmatic Usage

``` python
from query_engine import SimpleRAGQuery

rag = SimpleRAGQuery()
result = rag.query("What caused the patient's readmission?", top_k=5)

print(result["answer"])
print(result["sources"])
```

------------------------------------------------------------------------

## 📁 Project Structure

    medical-rag-system/
    │
    ├── add_embeddings.py
    ├── build_faiss_index.py
    ├── entity_extraction.py
    ├── kg_extraction.py
    ├── kg_rag_query.py
    ├── process_batch.py
    ├── config.py
    │
    ├── data_config.json
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🔧 Configuration Options

### Chunking

``` python
CHUNK_SETTINGS = {
  "max_tokens": 500,
  "overlap": 50,
  "min_chunk_size": 50
}
```

### Embeddings

-   **OpenAI** (highest quality)
-   **Sentence-Transformers local model** (no cost)

### LLM Options

-   **Llama 3.2 3B** (default)
-   **Llama 3.1 8B** (more accurate)
-   **BioMistral 7B** (medical-tuned)

------------------------------------------------------------------------

## 📈 Performance Benchmarks

### Processing (M1 MacBook Pro)

  Stage                Time
  -------------------- -------------------
  Parsing + Chunking   \<1 sec/note
  Entity Extraction    3--5 sec/chunk
  Embeddings (local)   \~1 sec/chunk
  **Total**            \~45--60 sec/note

### Query Performance

  Component        Latency
  ---------------- ---------
  Vector Search    \<100ms
  Reranking        \~50ms
  LLM Generation   3--8s
  **Total**        4--9s

------------------------------------------------------------------------

## 🗄️ MongoDB Schema (processed_chunks)

``` json
{
  "chunk_id": "note001_hpi_0",
  "note_id": "note001",
  "section": "hpi",
  "text": "...",
  "context_prefix": "[Section: HPI | Temporal: Admission]",
  "full_text": "[prefix]
---
[text]",
  "embedding": [...],
  "metadata": {
    "temporal_marker": "admission",
    "entities": {...}
  },
  "chunk_index": 0
}
```

------------------------------------------------------------------------

## 🔗 Neo4j Graph Schema

### Node Types

-   `Note`
-   `Chunk`
-   `Diagnosis`
-   `Medication`
-   `Symptom`
-   `Procedure`

### Relationships

-   `HAS_CHUNK`
-   `HAS_DIAGNOSIS`
-   `TREATED_WITH`
-   `CAUSES`
-   `DOCUMENTS`

------------------------------------------------------------------------

## 🔐 Privacy & Compliance

-   Fully local execution (MongoDB, Neo4j, LLM, embeddings)
-   Compatible with de-identified MIMIC data
-   No PHI sent to external APIs (unless OpenAI embeddings are enabled)
-   Encryption-enabled Atlas support

------------------------------------------------------------------------

## 🐛 Troubleshooting

### MongoDB connection

``` bash
mongosh $MONGODB_URI
```

### Neo4j authentication

``` bash
docker restart neo4j-medical
```

### Ollama issues

``` bash
ollama serve
ollama list
```

### Missing embeddings

``` bash
python scripts/check_embeddings.py
```

------------------------------------------------------------------------

## 🗺 Roadmap

### **v0.1 -- Completed**

-   Ingestion pipeline\
-   MongoDB + Neo4j integration\
-   Local LLM inference\
-   Vector search retrieval

### **v0.2 -- In Progress**

-   Cross-encoder reranking\
-   BioMistral integration\
-   Streamlit web UI\
-   Query-routing (agentic RAG)

### **v0.3+ -- Planned**

-   Fine-tuned clinical embeddings\
-   Real-time indexing\
-   Multi-patient analytics\
-   Temporal graph reasoning

------------------------------------------------------------------------

## 🤝 Contributing

1.  Fork\
2.  Create branch\
3.  Commit\
4.  Push\
5.  Open PR

See `CONTRIBUTING.md`.

------------------------------------------------------------------------

## 📄 License

MIT License.\
MIMIC-IV data requires PhysioNet credentialed access.

------------------------------------------------------------------------

## ⭐ Star the Project

If this helps your research or clinical NLP work, please consider
starring the repo!
