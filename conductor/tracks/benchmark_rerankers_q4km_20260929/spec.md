# Specification: Benchmark Científico de Rerankers em Pipeline RAG Industrial (Foco Q4_K_M)

## 1. Overview & Contexto
Esta track tem como objetivo implementar, validar e comparar experimentalmente modelos de re-ranking neural em dois estágios (*two-stage retrieval*) sobre os 16 manuais técnicos industriais de nobreaks da CM Comandos Lineares. O re-ranking atua reordenando os $K_{cand} = 20$ candidatos recuperados pelos dois modelos campeões de embedding do benchmark anterior:
1. `qwen3_8b_q4km` (Melhor cobertura de nós relevantes, $HR@5 = 0.5703$, $Recall@5 = 0.5311$)
2. `multilingual_e5_large` (Melhor precisão de primeiro posto, $MRR@5 = 0.4031$)

O foco central da pesquisa é avaliar a viabilidade e os ganhos de acurácia de modelos de re-ranking generativos e discriminativos quantizados em 4 bits (`Q4_K_M`) operando sob restrição de **8 GB de VRAM** (GPU NVIDIA RTX PRO 1000).

## 2. Hipóteses Científicas
- **Hipótese Nula ($H_0$)**: A introdução de re-ranking neural quantizado em 4 bits (`Q4_K_M`) não produz ganhos estatisticamente significantes em relação à recuperação densa isolada ($MRR@5_{\text{rerank}} \le MRR@5_{\text{puro}}$ com $p \ge 0.05$ no teste pareado de Wilcoxon).
- **Hipótese Alternativa ($H_1$)**: O modelo `Qwen3-Reranker-4B:Q4_K_M` eleva significativamente o $MRR@5$ e o $HitRate@5$ sobre a recuperação pura ($p < 0.05$), atingindo desempenho competitivo ou superior ao baseline proprietário `RankGPT (gpt-4o-mini)` com latência $\le 100\text{ ms}$ e consumo de VRAM $\le 3.5\text{ GB}$.

## 3. Bancada de Modelos de Re-ranking
1. **`dengcao/Qwen3-Reranker-4B:Q4_K_M`** (Ollama Local, ~2.5 GB VRAM): Re-ranking generativo pontuado via logprob softmax binário (`yes`/`no`) utilizando o endpoint nativo `/api/chat` com `think: false`, `num_predict: 1`.
2. **`dengcao/Qwen3-Reranker-8B:Q5_K_M` ou `Q4_K_M`** (Ollama Local, ~5.8 GB / ~4.8 GB): Re-ranking generativo de escala 8B para avaliar a lei de escala em re-ranking.
3. **`BAAI/bge-reranker-v2-m3`** (HuggingFace / Ollama, ~1.1 GB): Cross-encoder discriminativo multilíngue de referência da indústria.
4. **`RankGPT (gpt-4o-mini)`** (OpenAI API / LlamaIndex): Baseline comercial proprietário idêntico ao utilizado no repositório `cm-comandos-piad`.
5. **Baseline de Controle (Puro)**: Recuperação vetorial densa sem re-ranking ($K_{cand} \rightarrow Top\text{-}K$).

## 4. Requisitos Funcionais
- **RF01**: Módulo de abstração unificado `scripts/rerankers.py` implementando interface comum (`rerank(query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]`).
- **RF02**: Adaptador `Qwen3OllamaReranker` com extração precisa de logprobs de `"yes"` e `"no"` e desativação estrita do modo reasoning (`think: false`).
- **RF03**: Adaptador `CrossEncoderReranker` com suporte a modelos discriminativos via PyTorch/CUDA.
- **RF04**: Adaptador `RankGPTReranker` com integração nativa ao `llama-index-postprocessor-rankgpt-rerank`.
- **RF05**: Pipeline de benchmark `scripts/benchmark_rerankers.py` que executa a matriz completa: 2 Embedders $\times$ 5 Estratégias de Rerank $\times$ 128 Consultas deduplicadas.
- **RF06**: Integração com o `matching_engine.py` (3 camadas: substring, RapidFuzz $\ge 85\%$, ROUGE-L $\ge 0.50$) para computar $HR@K$, $MRR@K$, $Recall@K$, $MAP@K$ para $K \in \{2, 5, 10\}$.
- **RF07**: Consolidação estatística (`scripts/consolidate_rerankers.py`) calculando ganhos delta ($\Delta$), teste pareado de Wilcoxon e Bootstrap 95% CI.
- **RF08**: Geração de gráficos científicos (300 DPI) em `graficos_tcc/`:
  - Comparação $\Delta MRR@5$ e $\Delta HR@5$ por reranker
  - Trade-off Latência incremental vs. Ganho de Acurácia
  - Consumo de VRAM e QPS

## 5. Requisitos Não-Funcionais & Restrições de Engenharia
- **RNF01 (Hardware)**: Execução estritamente sequencial na GPU NVIDIA RTX PRO 1000 (8 GB VRAM), com descarga compulsória de memória entre modelos (`keep_alive=0`, `torch.cuda.empty_cache()`, `gc.collect()`).
- **RNF02 (Determinismo)**: `SEED = 42` fixada em todas as etapas amostrais.
- **RNF03 (Degradação Graciosa)**: Fallback seguro para a ordem original pré-rerank em caso de falha de conexão ou timeout.
- **RNF04 (Testes Unitários)**: Cobertura mínima de 80% nos testes dos adaptadores de reranker e cálculo de softmax.

## 6. Critérios de Aceite
1. Todos os testes unitários em `tests/test_rerankers.py` passando com 100% de sucesso.
2. Execução da matriz completa de benchmark (10 configurações $\times$ 128 queries) sem estouro de VRAM (*Out of Memory*).
3. Geração dos arquivos de dados brutos (`results/raw_rerank_*.json`) e CSV consolidado (`results/consolidated_reranker_metrics.csv`).
4. Gráficos científicos salvos em `graficos_tcc/`.
5. Relatório comparativo com análise de confirmação/rejeição de $H_1$ atualizado.
