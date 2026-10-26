# Avaliação RAGAS — gpt-6-luna com Diferentes Níveis de Reasoning Effort

## Metodologia

- **Modelo Gerador:** `gpt-6-luna` (via 9router / Codex)
- **Judge:** `gpt-6-astra` com `reasoning_effort=high` (via 9router / Codex)
- **Embeddings:** `openai/text-embedding-3-small` (via OpenRouter)
- **Dataset:** `5` amostras industriais extraídas de manuais técnicos (`testset_openai_4omini.jsonl`)
- **Métricas RAGAS:** Faithfulness (fidelidade ao contexto recuperado) e Answer Relevancy (aderência semântica à pergunta)

## Resultados Comparativos Consolidados

| Reasoning Effort | Faithfulness | Answer Relevancy | Amostras |
| :--- | :---: | :---: | :---: |
| **none** | 0.3500 | 0.7873 | 5 |
| **low** | 0.3600 | 0.7729 | 5 |
| **medium** | 0.4000 | 0.7656 | 5 |
| **high** | 0.4000 | 0.8111 | 5 |

## Conclusões e Análise dos Níveis de Esforço

1. **none (sem reasoning):** Mantém latência mínima e bom alinhamento com contextos diretos.
2. **low / medium / high:** Níveis de esforço cognitivo progressivos permitindo raciocínio analítico mais aprofundado.
3. **Judge Rigoroso:** O uso do `gpt-6-astra` com `high` effort atua como avaliador neutro com reasoning extensivo para verificar fidelidade e relevância.