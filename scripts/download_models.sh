#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Download e Cache de Modelos de Embedding (CUDA / GPU)"
echo "=========================================================="

echo "[1/2] Verificando e baixando modelos no Ollama..."
echo "-> qwen3-embedding:8b (Q4_K_M)..."
ollama pull qwen3-embedding:8b

echo "-> qwen3-embedding:4b (Q4_K_M)..."
ollama pull qwen3-embedding:4b

echo "-> bge-m3 (Multilíngue)..."
ollama pull bge-m3

echo "-> nomic-embed-text (Ultraleve)..."
ollama pull nomic-embed-text

echo "[2/2] Pré-carregando modelos HuggingFace no cache local..."
.venv/bin/python -c "
from sentence_transformers import SentenceTransformer
print('-> Baixando/Verificando intfloat/multilingual-e5-large...')
SentenceTransformer('intfloat/multilingual-e5-large')
print('-> Baixando/Verificando ricardoz/bertimbau-base-portuguese-sts...')
SentenceTransformer('ricardoz/bertimbau-base-portuguese-sts')
print('Modelos HuggingFace prontos em cache!')
"

echo "=========================================================="
echo "  Todos os modelos foram baixados e cacheados com sucesso!"
echo "=========================================================="
