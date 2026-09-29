# Tabela Resumo do Benchmark de Re-ranking Neural (Two-Stage RAG)

Avaliação do impacto de reordenadores neurais nos Top-20 candidatos recuperados pelos melhores modelos de embedding.

| Modelo Base | Reranker | HR@5 | MRR@5 | Δ MRR@5 | Δ HR@5 | Wilcoxon (p) | Latência Re-rank (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `multilingual_e5_large` | `none` | **54.7%** | **0.4031** | - | - | Baseline | 0.0 ms |
| `multilingual_e5_large` | `bge_reranker_v2_m3` | **57.0%** | **0.4982** | +0.0951 | +2.3% | 0.0261 | 1164.3 ms |
| `multilingual_e5_large` | `qwen3_4b_q4km` | **62.5%** | **0.4764** | +0.0733 | +7.8% | 0.0406 | 6211.0 ms |
| `multilingual_e5_large` | `qwen3_8b_q5km` | **65.6%** | **0.4975** | +0.0944 | +10.9% | 0.0023 | 6894.9 ms |
| `multilingual_e5_large` | `rankgpt` | **61.7%** | **0.4573** | +0.0542 | +7.0% | 0.0510 | 1723.0 ms |
| `qwen3_8b_q4km` | `none` | **57.0%** | **0.3887** | - | - | Baseline | 0.0 ms |
| `qwen3_8b_q4km` | `bge_reranker_v2_m3` | **53.9%** | **0.4866** | +0.0979 | -3.1% | 0.0038 | 1145.1 ms |
| `qwen3_8b_q4km` | `qwen3_4b_q4km` | **60.9%** | **0.4738** | +0.0852 | +3.9% | 0.0049 | 5181.6 ms |
| `qwen3_8b_q4km` | `qwen3_8b_q5km` | **60.2%** | **0.5029** | +0.1142 | +3.1% | 0.0001 | 6424.2 ms |
| `qwen3_8b_q4km` | `rankgpt` | **57.8%** | **0.4663** | +0.0776 | +0.8% | 0.0160 | 1647.8 ms |

*Nota: Teste de Wilcoxon pareado calculado sobre o ganho de MRR@5 consulta por consulta contra o Baseline Puro (`none`).*
