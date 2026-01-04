# Graph RAG Pipeline - Local Deployment Runbook

This runbook provides step-by-step instructions for running the Graph RAG system locally.

## Overview

This Graph RAG (Retrieval-Augmented Generation) system is designed for clinical decision support using medical notes from the MIMIC dataset. It combines:

- **MongoDB**: Document store for notes and processed chunks
- **Neo4j**: Knowledge graph for concept relationships
- **FAISS**: Vector similarity search for embeddings
- **BioClinicalBERT**: Medical domain embeddings
- **Ollama**: Local LLM for answer generation

### Architecture

```
MIMIC.filtered_notes.json
         │
         ▼
┌─────────────────────────────────┐
│    MongoDB: filtered_notes      │ ◄── Raw clinical notes
└─────────────────────────────────┘
         │
         ▼ process_batch.py
┌─────────────────────────────────┐
│   MongoDB: processed_chunks     │ ◄── Chunked notes with metadata
└─────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
add_embeddings.py    ──► MongoDB (embedding field)
build_faiss_index.py ──► data/faiss_biobert.index
kg_extraction.py     ──► Neo4j (Chunk→Concept graph)
         │
         ▼
┌─────────────────────────────────┐
│      kg_rag_query.py            │
│  1. Embed query (BioClinicalBERT)
│  2. FAISS similarity search
│  3. Fetch docs from MongoDB
│  4. Hybrid rank with KG concepts
│  5. Generate answer (Ollama LLM)
└─────────────────────────────────┘
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Pipeline scripts |
| MongoDB | 6.0+ | Document storage |
| Neo4j | 5.0+ | Knowledge graph |
| Ollama | Latest | Local LLM |

### Hardware Requirements

- **RAM**: Minimum 16GB (32GB recommended for embeddings)
- **Disk**: ~5GB for models + data
- **GPU**: Optional (CPU works but slower for embeddings)

---

## 1. Setup

### 1.1 Clone and Enter Directory

```bash
cd /path/to/GraphRAG
```

### 1.2 Python Environment Setup

Choose ONE of the following options:

---

#### Option A: Conda Setup (Recommended for GPU)

This is the recommended approach, especially if you have a GPU.

**Automated Setup (Linux/Mac):**
```bash
bash setup_conda.sh
```

**Automated Setup (Windows):**
```cmd
setup_conda.bat
```

**Manual Conda Setup:**
```bash
# Create environment
conda create -n graphrag python=3.10 -y
conda activate graphrag

# Install PyTorch with CUDA (adjust cuda version: 11.8 or 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CPU-only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install core packages
conda install numpy=1.26.4 scipy scikit-learn -c conda-forge -y
conda install spacy=3.4.4 -c conda-forge -y
conda install pymongo tqdm pyyaml requests python-dotenv -c conda-forge -y

# Install pip packages
pip install sentence-transformers>=2.2.0 transformers>=4.30.0 huggingface-hub>=0.14.0
pip install faiss-gpu  # or faiss-cpu for CPU-only
pip install neo4j>=5.0.0
pip install scispacy==0.5.4

# Install spaCy medical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

**Verify GPU is working:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

#### Option B: Pip/Venv Setup (CPU or simple setup)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Install spaCy model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

---

### 1.3 Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# MongoDB
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=MIMIC

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# LLM (Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

---

## 2. Start External Services

### 2.1 Start MongoDB

**Option A: Direct install**
```bash
# Start MongoDB daemon
mongod --dbpath /path/to/data/db

# Or if using systemd
sudo systemctl start mongod
```

**Option B: Docker**
```bash
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  mongo:6.0
```

**Verify MongoDB is running:**
```bash
mongosh --eval "db.adminCommand('ping')"
# Should return: { ok: 1 }
```

### 2.2 Start Neo4j

**Option A: Direct install**
```bash
neo4j start
```

**Option B: Docker**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password_here \
  -v neo4j_data:/data \
  neo4j:5
```

**Verify Neo4j is running:**
- Open http://localhost:7474 in browser
- Login with credentials from `.env`

### 2.3 Start Ollama

```bash
# Start Ollama service
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:3b

# Alternative: medical-focused model
ollama pull meditron:7b
```

**Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

---

## 3. Load MIMIC Data

### 3.1 Place Your Data File

Ensure `MIMIC.filtered_notes.json` is in the project root (or specify path).

The file should contain clinical notes in one of these formats:

**JSON Array format:**
```json
[
  {"subject_id": 10001, "hadm_id": 20001, "text": "Patient presents with..."},
  {"subject_id": 10002, "hadm_id": 20002, "text": "Chief complaint: ..."}
]
```

**NDJSON format (one object per line):**
```json
{"subject_id": 10001, "hadm_id": 20001, "text": "Patient presents with..."}
{"subject_id": 10002, "hadm_id": 20002, "text": "Chief complaint: ..."}
```

### 3.2 Import Notes into MongoDB

```bash
python import_mimic_notes.py --file ./MIMIC.filtered_notes.json
```

**Options:**
- `--file PATH`: Path to JSON file (default: `./MIMIC.filtered_notes.json`)
- `--drop`: Drop existing collection before import

**Expected output:**
```
============================================================
MIMIC Notes Import Script
============================================================
File:       ./MIMIC.filtered_notes.json
MongoDB:    mongodb://localhost:27017/
Database:   MIMIC
Collection: filtered_notes
============================================================

Detecting file format...
Format: array
Loading documents from file...
Loaded 500 documents from file.
Validating and normalizing documents...
Valid documents: 500
Inserting documents into MongoDB...
Creating indexes...

============================================================
IMPORT COMPLETE
============================================================
Documents inserted: 500
Total in collection: 500
Unique patients: 50
Unique admissions: 150
============================================================
```

### 3.3 Verify Import

```bash
mongosh MIMIC --eval "db.filtered_notes.countDocuments({})"
# Should return number of imported notes

mongosh MIMIC --eval "db.filtered_notes.findOne()"
# Should show a sample document
```

---

## 4. Build the Graph RAG Index

### 4.1 Process Notes into Chunks

This script:
- Reads notes from `filtered_notes` collection
- Parses sections (Chief Complaint, HPI, Physical Exam, etc.)
- Creates smart chunks (respecting sentence boundaries)
- Stores chunks in `processed_chunks` collection
- Creates Note and Chunk nodes in Neo4j

```bash
python process_batch.py
```

When prompted for batch size, enter a number (default: 10) or press Enter.

**Expected output:**
```
================= Batch Pipeline Ready =================
MongoDB       -> mongodb://localhost:27017/
Database      -> MIMIC
Source Notes  -> filtered_notes
Output Chunks -> processed_chunks
Neo4j         -> bolt://localhost:7687
========================================================

Total notes available: 500
Already processed: 0

Starting batch of 10 notes...
Processing: 100%|████████████████████| 10/10 [00:05<00:00]

================ BATCH COMPLETE ================
  Success     -> 10
  Failed      -> 0
  Chunks Made -> 87
================================================
```

**Run multiple batches** to process all notes:
```bash
# Process all notes in batches of 100
python -c "
from process_batch import BatchProcessor
p = BatchProcessor()
while True:
    stats = p.process_batch(100)
    if stats['success'] == 0:
        break
p.cleanup()
"
```

### 4.2 Add Embeddings

This script adds BioClinicalBERT embeddings to each chunk.

```bash
python add_embeddings.py
```

**Note:** First run downloads the model (~400MB). Embedding is CPU-bound and may take 1-2 minutes per 100 chunks.

**Expected output:**
```
============================================================
Adding BioClinicalBERT embeddings to processed_chunks
============================================================
MongoDB: mongodb://localhost:27017/
Database: MIMIC
Collection: processed_chunks
Model: emilyalsentzer/Bio_ClinicalBERT
Embedding dimension: 768
Chunks missing embeddings: 87
Embedding chunks: 100%|████████████████| 87/87 [00:45<00:00]

============================================================
Finished embedding processed_chunks
============================================================
```

### 4.3 Build FAISS Index

This creates the vector search index.

```bash
python build_faiss_index.py
```

**Expected output:**
```
============================================================
Building FAISS index from processed_chunks.embeddings
============================================================
Chunks with embeddings: 87
Collecting vectors: 100%|████████████████| 87/87
Matrix shape: (87, 768)
FAISS index built with 87 vectors, dim=768
Saved index -> data/faiss_biobert.index
Saved mapping -> data/faiss_biobert_mapping.json
============================================================
```

### 4.4 Extract Knowledge Graph (Optional but Recommended)

This extracts medical concepts using spaCy NER and creates Concept nodes in Neo4j.

```bash
python kg_extraction.py
```

**Expected output:**
```
============================================================
Running KG Extraction over processed_chunks -> Neo4j
============================================================
Chunks to process (kg_status != 'done'): 87
KG extracting: 100%|████████████████| 87/87 [00:30<00:00]

================ KG EXTRACTION COMPLETE ================
  Processed chunks : 87
  Failed chunks    : 0
========================================================
```

---

## 5. Query the System

### 5.1 Interactive Query Mode

```bash
python kg_rag_query.py
```

**Example session:**
```
============================================================
KG-AWARE MEDICAL RAG SYSTEM (Hybrid KG Mode)
============================================================

Enter your medical question:
> What medications was the patient on for heart failure?

Filter by patient_id (subject_id) [blank = all]: 10001
Filter by hadm_id [blank = all]:

============================================================
KG-AWARE MEDICAL RAG QUERY (Hybrid KG Mode)
============================================================
MongoDB:   mongodb://localhost:27017/  DB=MIMIC  Coll=processed_chunks
Neo4j:     bolt://localhost:7687
FAISS:     data/faiss_biobert.index
LLM:       llama3.2:3b
============================================================

Loading embedding model -> emilyalsentzer/Bio_ClinicalBERT
FAISS Loaded | dim=768 | entries=87

QUESTION: What medications was the patient on for heart failure?
Filter: subject_id=10001 | hadm_id=None

FAISS neighbours (index -> score):
   * 42 -> chunk_id=..., score=0.8234
   ...

Using top 5 chunks after hybrid scoring.

================= FINAL ANSWER =================

1. Short Answer
The patient was prescribed Lisinopril 10mg daily, Metoprolol 25mg twice daily,
and Furosemide 40mg daily for heart failure management...

2. Reasoning
- Evidence shows ACE inhibitor (Lisinopril) for afterload reduction
- Beta blocker (Metoprolol) for rate control and mortality benefit
- Loop diuretic (Furosemide) for volume management
...

3. Evidence Summary
- [chunk:abc123_meds_discharge_15] - Discharge medications list
- [chunk:abc123_hospital_course_8] - Treatment during admission
...

================================================
```

### 5.2 Programmatic Usage

```python
from kg_rag_query import kg_rag_query

answer = kg_rag_query(
    question="What were the patient's vital signs on admission?",
    top_k=5,
    subject_id=10001,  # Optional: filter by patient
    hadm_id=20001      # Optional: filter by admission
)
```

### 5.3 Multi-Hop Knowledge Graph Queries

The KG integration enables reasoning across multiple concepts. Test these complex queries to verify the system synthesizes information from graph paths correctly.

#### Example 1: Disease → Treatment → Complication

```bash
python kg_rag_query.py
```

**Query:**
```
What medications were used to treat heart failure in patients who also developed acute kidney injury?
```

This tests: `Heart Failure → Treatment → Drug → Complication → AKI`

**Expected behavior:**
- Retrieves chunks mentioning both heart failure treatment AND kidney complications
- KG hybrid scoring should boost chunks with overlapping concepts
- Answer should cite specific drugs (e.g., diuretics) and mention renal monitoring

---

#### Example 2: Symptom → Diagnosis → Treatment Chain

**Query:**
```
For patients presenting with shortness of breath who were diagnosed with pneumonia, what antibiotics were prescribed?
```

This tests: `Symptom (dyspnea) → Diagnosis (pneumonia) → Treatment (antibiotics)`

**Expected reasoning chain:**
1. Initial presentation: dyspnea/SOB
2. Workup leading to pneumonia diagnosis
3. Antibiotic selection based on clinical context

---

#### Example 3: Procedure → Complication → Management

**Query:**
```
What complications occurred after cardiac catheterization and how were they managed?
```

This tests: `Procedure → Complication → Intervention`

---

#### Example 4: Temporal Multi-Hop (Admission → Hospital Course → Discharge)

**Query:**
```
How did the patient's blood pressure management change from admission to discharge?
```

This tests the system's ability to:
- Distinguish temporal stages (admission vs discharge)
- Track medication changes over time
- Synthesize information across multiple note sections

---

#### Example 5: Complex Clinical Reasoning

**Query:**
```
In patients with diabetes who developed sepsis, what insulin protocols were used and did they require vasopressor support?
```

This tests: `Comorbidity (DM) → Complication (sepsis) → Treatment (insulin) → Additional intervention (vasopressors)`

---

### 5.4 Verifying KG Contribution to Answers

To see how much the Knowledge Graph contributes to retrieval quality, compare with and without KG:

```python
from kg_rag_query import (
    init_mongo, init_faiss, init_embedder, init_neo4j,
    embed_query, faiss_candidates, attach_docs_and_filter,
    hybrid_rank, build_full_context
)

# Initialize components
mongo_client, coll = init_mongo()
index, mapping = init_faiss()
embedder = init_embedder()
driver = init_neo4j()

question = "What antibiotics were used for pneumonia with renal impairment?"

# Get embedding
q_vec = embed_query(embedder, question)

# FAISS-only retrieval
faiss_hits = faiss_candidates(index, mapping, q_vec, oversample=50)
filtered = attach_docs_and_filter(faiss_hits, coll, top_k=10, subject_id=None, hadm_id=None)

# Hybrid retrieval (FAISS + KG)
hybrid_results = hybrid_rank(filtered, driver, top_k=5)

# Compare scores
print("\n=== FAISS-only top 5 ===")
for i, r in enumerate(filtered[:5]):
    print(f"{i+1}. {r['chunk_id']} | FAISS: {r['faiss_score']:.4f}")

print("\n=== Hybrid (FAISS + KG) top 5 ===")
for i, r in enumerate(hybrid_results):
    print(f"{i+1}. {r['chunk_id']} | Hybrid: {r['hybrid_score']:.4f} | KG: {r['kg_score']:.4f} | Concepts: {len(r['concepts'])}")

mongo_client.close()
driver.close()
```

**What to look for:**
- Hybrid ranking should reorder results based on concept density
- Chunks with more medical concepts should rank higher
- Multi-hop queries should benefit most from KG boosting

---

### 5.5 Neo4j Graph Exploration

Explore the knowledge graph directly to understand concept relationships:

```cypher
// Find all concepts mentioned with a specific disease
MATCH (c:Chunk)-[:MENTIONS_CONCEPT]->(e:Concept)
WHERE e.name CONTAINS 'heart failure' OR e.name CONTAINS 'pneumonia'
RETURN e.name, count(c) AS chunk_count
ORDER BY chunk_count DESC
LIMIT 20;

// Find chunks that mention multiple related concepts
MATCH (c:Chunk)-[:MENTIONS_CONCEPT]->(e1:Concept),
      (c)-[:MENTIONS_CONCEPT]->(e2:Concept)
WHERE e1.name CONTAINS 'diabetes' AND e2.name CONTAINS 'insulin'
RETURN c.chunk_id, e1.name, e2.name;

// Find co-occurring concepts (potential relationships)
MATCH (c:Chunk)-[:MENTIONS_CONCEPT]->(e1:Concept),
      (c)-[:MENTIONS_CONCEPT]->(e2:Concept)
WHERE e1 <> e2
RETURN e1.name, e2.name, count(c) AS co_occurrence
ORDER BY co_occurrence DESC
LIMIT 30;

// Trace path from symptom to treatment
MATCH path = (c1:Chunk)-[:MENTIONS_CONCEPT]->(symptom:Concept),
             (c2:Chunk)-[:MENTIONS_CONCEPT]->(treatment:Concept)
WHERE symptom.name CONTAINS 'fever' AND treatment.name CONTAINS 'antibiotic'
  AND c1.note_id = c2.note_id
RETURN symptom.name, treatment.name, c1.section, c2.section
LIMIT 10;
```

---

## 6. Validation and Sanity Checks

### 6.1 Check MongoDB Collections

```bash
mongosh MIMIC --eval "
print('Notes:', db.filtered_notes.countDocuments({}));
print('Chunks:', db.processed_chunks.countDocuments({}));
print('Chunks with embeddings:', db.processed_chunks.countDocuments({embedding: {\$exists: true}}));
"
```

### 6.2 Check Neo4j Graph

Open Neo4j Browser (http://localhost:7474) and run:

```cypher
// Count nodes
MATCH (n:Note) RETURN count(n) AS notes;
MATCH (c:Chunk) RETURN count(c) AS chunks;
MATCH (e:Concept) RETURN count(e) AS concepts;

// Sample relationships
MATCH (n:Note)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS_CONCEPT]->(e:Concept)
RETURN n.note_id, c.chunk_id, e.name
LIMIT 10;
```

### 6.3 Check FAISS Index

```python
import faiss
import json

index = faiss.read_index("data/faiss_biobert.index")
print(f"FAISS vectors: {index.ntotal}")
print(f"Dimensions: {index.d}")

with open("data/faiss_biobert_mapping.json") as f:
    mapping = json.load(f)
print(f"Mapping entries: {len(mapping)}")
```

### 6.4 Test Query Pipeline

```bash
# Quick test query
python -c "
from kg_rag_query import kg_rag_query
kg_rag_query('What is the patient diagnosis?', top_k=3)
"
```

---

## 7. Troubleshooting

### MongoDB Connection Errors

**Error:** `ServerSelectionTimeoutError: localhost:27017`

**Solution:**
1. Check MongoDB is running: `mongosh --eval "db.adminCommand('ping')"`
2. Verify `MONGODB_URI` in `.env`
3. Check firewall/port access

### Neo4j Connection Errors

**Error:** `ServiceUnavailable: Unable to establish connection`

**Solution:**
1. Check Neo4j is running: open http://localhost:7474
2. Verify `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` in `.env`
3. Ensure bolt port (7687) is accessible

### FAISS Index Not Found

**Error:** `FileNotFoundError: Missing FAISS index at data/faiss_biobert.index`

**Solution:**
1. Run `python add_embeddings.py` first
2. Then run `python build_faiss_index.py`

### Empty Query Results

**Symptoms:** "No FAISS hits for this query" or "No chunks matched filters"

**Solutions:**
1. Verify data was imported: `mongosh MIMIC --eval "db.processed_chunks.countDocuments({})"`
2. Check FAISS index was built: `ls -la data/faiss_biobert.index`
3. Try query without patient/admission filters
4. Ensure embeddings exist: `mongosh MIMIC --eval "db.processed_chunks.countDocuments({embedding: {\$exists: true}})"`

### Out of Memory During Embeddings

**Solution:**
- Process in smaller batches
- Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`
- Increase swap space

### Ollama Not Responding

**Error:** `requests.exceptions.ConnectionError`

**Solution:**
1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Verify model is downloaded: `ollama list`
3. Pull model if missing: `ollama pull llama3.2:3b`

---

## 8. Maintenance

### Reset KG Extraction Status

To re-run KG extraction on all chunks:

```bash
python reset_kg_flags.py
python kg_extraction.py
```

### Rebuild FAISS Index

If embeddings change or get corrupted:

```bash
rm data/faiss_biobert.index data/faiss_biobert_mapping.json
python build_faiss_index.py
```

### Clear All Data and Start Fresh

```bash
# MongoDB
mongosh MIMIC --eval "db.filtered_notes.drop(); db.processed_chunks.drop(); db.processing_log.drop();"

# Neo4j (in Neo4j Browser)
MATCH (n) DETACH DELETE n;

# FAISS
rm -rf data/faiss*.index data/faiss*.json

# Re-run pipeline
python import_mimic_notes.py --file MIMIC.filtered_notes.json
python process_batch.py  # Run multiple times for all notes
python add_embeddings.py
python build_faiss_index.py
python kg_extraction.py
```

---

## Quick Reference

| Step | Command | Description |
|------|---------|-------------|
| 1 | `source venv/bin/activate` | Activate Python environment |
| 2 | `mongod` | Start MongoDB |
| 3 | `neo4j start` | Start Neo4j |
| 4 | `ollama serve` | Start Ollama |
| 5 | `python import_mimic_notes.py` | Import notes to MongoDB |
| 6 | `python process_batch.py` | Process notes into chunks |
| 7 | `python add_embeddings.py` | Generate embeddings |
| 8 | `python build_faiss_index.py` | Build FAISS index |
| 9 | `python kg_extraction.py` | Extract KG (optional) |
| 10 | `python kg_rag_query.py` | Run queries |
