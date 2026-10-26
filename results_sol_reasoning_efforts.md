# Avaliação RAGAS — gpt-6-sol com Diferentes Níveis de Reasoning Effort

## Metodologia

- **Modelo Gerador:** `gpt-6-sol` (via 9router / Codex)
- **Judge:** `gpt-6-astra` com `reasoning_effort=high` (via 9router / Codex)
- **Embeddings:** `openai/text-embedding-3-small` (via OpenRouter)
- **Dataset:** `5` amostras industriais extraídas de manuais técnicos (`testset_openai_4omini.jsonl`)

## Resultados Comparativos Consolidados (gpt-6-sol)

| Modelo | Reasoning Effort | Faithfulness | Answer Relevancy | Amostras |
| :--- | :--- | :---: | :---: | :---: |
| **gpt-6-sol** | **none** | 0.3667 | 0.7987 | 5 |
| **gpt-6-sol** | **low** | 0.3667 | 0.8120 | 5 |
| **gpt-6-sol** | **medium** | 0.4000 | 0.8114 | 5 |
| **gpt-6-sol** | **high** | 0.4000 | 0.7824 | 5 |