#!/bin/bash
# Graph RAG - Conda Environment Setup Script
# Usage: bash setup_conda.sh

set -e

ENV_NAME="graphrag"
CUDA_VERSION="11.8"  # Change to 12.1 if you have newer CUDA

echo "============================================================"
echo "Graph RAG - Conda Environment Setup"
echo "============================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Installing CPU-only version."
    CUDA_VERSION="cpu"
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing '${ENV_NAME}' environment..."
    conda env remove -n ${ENV_NAME} -y
fi

echo ""
echo "Creating conda environment '${ENV_NAME}'..."
echo ""

# Create environment with core packages
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
if [ "$CUDA_VERSION" = "cpu" ]; then
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
else
    conda install pytorch torchvision torchaudio pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y
fi

echo "Installing core dependencies..."
conda install numpy=1.26.4 scipy scikit-learn -c conda-forge -y

echo "Installing spaCy 3.4.4..."
conda install spacy=3.4.4 -c conda-forge -y

echo "Installing other conda packages..."
conda install pymongo tqdm pyyaml requests python-dotenv -c conda-forge -y

echo "Installing pip packages..."
pip install --upgrade pip

# Install transformers ecosystem
pip install sentence-transformers>=2.2.0 transformers>=4.30.0 huggingface-hub>=0.14.0

# Install FAISS (GPU or CPU)
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install faiss-cpu
else
    pip install faiss-gpu
fi

# Install Neo4j driver
pip install neo4j>=5.0.0

# Install scispacy (compatible with spacy 3.4.4)
pip install scispacy==0.5.4

echo ""
echo "Installing SciSpaCy medical NER model..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz || {
    echo "WARNING: SciSpaCy model failed. Installing fallback model..."
    python -m spacy download en_core_web_sm
}

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To verify GPU support:"
echo "    python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')\""
echo ""
echo "Next steps:"
echo "    1. Copy .env.example to .env and configure"
echo "    2. Start MongoDB and Neo4j"
echo "    3. Run: python import_mimic_notes.py"
echo "    4. See RUNBOOK.md for full instructions"
echo "============================================================"
