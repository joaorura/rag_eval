# Technology Stack: CM Comandos RAG Benchmark

## Core Language & Runtime
- **Python**: 3.12 (gerenciador de ambiente e dependências via `uv`)
- **Sistema Operacional & Aceleração**: Linux x86_64 com suporte a NVIDIA CUDA 12+ (GPU NVIDIA RTX PRO 1000 com 8 GB GDDR6 VRAM)

## RAG Framework & Orchestration
- **LlamaIndex**: `llama-index==0.12.10`
  - `llama-index-core`: Segmentação com `SentenceSplitter(chunk_size=512, chunk_overlap=50)`, vetores e abstração de retrievers.
  - `llama-index-embeddings-ollama`, `llama-index-embeddings-huggingface`, `llama-index-embeddings-openai`
  - `llama-index-llms-ollama`, `llama-index-llms-openai`

## Model Serving & Inference Runtimes
- **Ollama**: Servidor local de inferência acelerado por GPU (CUDA), servindo modelos quantizados em GGUF (`Q4_K_M`): `qwen3-embedding:8b`, `qwen3-embedding:4b`, `bge-m3`, `nomic-embed-text`.
- **PyTorch & HuggingFace SentenceTransformers**: `torch>=2.4.0`, `sentence-transformers>=3.0.0` para execução local direta com CUDA (`multilingual-e5-large` com prefixos assimétricos e `juridics/bertimbau-base-portuguese-sts-scale`).
- **OpenAI API**: `text-embedding-3-small` (baseline de referência) e `gpt-4o-mini`.

## Evaluation & Information Retrieval (IR) Engine
- **RapidFuzz**: `RapidFuzz==3.11.0` (`token_set_ratio`, `partial_ratio`)
- **ROUGE Score**: `rouge-score==0.1.2` (métrica ROUGE-L $F_1$)
- **Ragas**: `ragas==0.2.3` (com adaptações personalizadas em `ragas_custom/`)

## Scientific Computing, Statistics & Visualization
- **SciPy**: `scipy>=1.13.0` (teste de Wilcoxon para postos com sinais pareados)
- **NumPy & Pandas**: `numpy`, `pandas` (estruturas de dados e reamostragem Bootstrap $B=1.000$)
- **Matplotlib & Seaborn**: `matplotlib>=3.9.0`, `seaborn==0.13.2` (geração de gráficos em 300 DPI)
- **Relatórios & Apresentações**: WeasyPrint 70.0 (PDF A4), Marp CLI (HTML/PDF slides), Python-PPTX (PowerPoint).

## Testing & Quality Assurance
- **Pytest / Unittest**: `pytest` com testes unitários em `tests/` para validação determinística de matching e auditoria.
