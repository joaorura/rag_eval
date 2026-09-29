# Tabela Resumo: Desempenho Comparativo de Recuperação (K=5)

| model_id              | quantization   |   hit_rate@5_mean |   mrr@5_mean |   recall@5_mean |   map@5_mean |   mean_latency_ms | wilcoxon_p_mrr5_vs_baseline   |
|:----------------------|:---------------|------------------:|-------------:|----------------:|-------------:|------------------:|:------------------------------|
| multilingual_e5_large | FP16/FP32      |            0.5469 |       0.4031 |          0.511  |       0.6523 |          22.8791  | 0.5217                        |
| qwen3_8b_q4km         | Q4_K_M         |            0.5703 |       0.3887 |          0.5311 |       0.6836 |          57.0615  | 0.9336                        |
| qwen3_4b_q4km         | Q4_K_M         |            0.5547 |       0.3844 |          0.4965 |       0.661  |          35.9795  | 0.9565                        |
| openai_3_small        | FP16/FP32      |            0.4688 |       0.384  |          0.4482 |       0.7286 |         415.074   | -                             |
| bge_m3                | FP16/FP32      |            0.5234 |       0.3797 |          0.4673 |       0.7441 |         167.296   | 0.8273                        |
| nomic_embed_text      | FP16/FP32      |            0.3594 |       0.3021 |          0.3466 |       0.5545 |           9.78261 | 0.0142                        |
| bertimbau_sts         | FP16/FP32      |            0.4531 |       0.2423 |          0.3498 |       0.362  |          35.6716  | 0.0008                        |

*Nota: Valores de p < 0.05 no teste de Wilcoxon indicam diferença estatisticamente significante em relação ao baseline da OpenAI.*
