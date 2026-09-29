# Implementation Plan: Benchmark Científico de Re-ranking Neural com foco em Q4_K_M

Track: [Specification](./spec.md) | [Metadata](./metadata.json)

## Phases

### Phase 1: Arquitetura Modular e Testes Unitários (TDD)
- [x] Task 1.1: Escrever testes unitários para os adaptadores de reranker e cálculo de logprob softmax (`tests/test_rerankers.py`).
- [x] Task 1.2: Implementar módulo de reranker unificado `scripts/rerankers.py`:
  - `Qwen3OllamaReranker`: chamada nativa ao Ollama com `think: false`, `num_predict: 1` e logprob softmax binário (`score = exp(lp_yes) / (exp(lp_yes) + exp(lp_no))`).
  - `CrossEncoderReranker`: suporte a `BAAI/bge-reranker-v2-m3` via `sentence-transformers` / HuggingFace.
  - `RankGPTReranker`: suporte a `RankGPTRerank` do LlamaIndex com `gpt-4o-mini`.
- [x] Task 1.3: Adicionar fallback determinístico (ordem original pré-rerank) e normalização de scores em `[0.0, 1.0]`.

### Phase 2: Pipeline de Benchmark de Re-ranking
- [x] Task 2.1: Criar `scripts/benchmark_rerankers.py` com suporte a super-sampling configurável ($K_{\text{cand}} = 20$).
- [x] Task 2.2: Implementar gestão sequencial estrita de VRAM (`keep_alive=0`, `torch.cuda.empty_cache()`, `gc.collect()`).
- [x] Task 2.3: Reutilizar o `matching_engine.py` (casamento em 3 camadas) para avaliar a reordenação contra as 128 perguntas de ground truth.

### Phase 3: Execução Experimental e Análise Estatística
- [x] Task 3.1: Executar matriz completa:
  - Base 1 (`qwen3_8b_q4km`) x [Puro, Qwen3-4B-Q4KM, Qwen3-8B-Q4KM/Q5KM, BGE-M3, RankGPT] (128 consultas cada).
  - Base 2 (`multilingual_e5_large`) x [Puro, Qwen3-4B-Q4KM] (128 consultas cada).
- [x] Task 3.2: Calcular ganhos incrementais ($\Delta$), teste pareado de Wilcoxon ($\alpha = 0.05$) e Bootstrap 95% ($B = 1.000$).
- [x] Task 3.3: Salvar resultados em `results/raw_rerank_*.json` e consolidar em `results/consolidated_reranker_metrics.csv`.

### Phase 4: Visualização e Documentação de Pesquisa
- [x] Task 4.1: Gerar gráficos comparativos em 300 DPI em `graficos_tcc/` (MRR@5 delta, Trade-off Acurácia vs Latência, Consumo de VRAM).
- [x] Task 4.2: Atualizar `results/summary_table_rerankers.md` e gerar síntese qualitativa dos ganhos de reranking para o relatório do TCC.
