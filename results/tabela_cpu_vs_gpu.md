# Comparativo de Latência de Inferência: CPU vs GPU

Avaliação empírica do tempo de recuperação por consulta (*end-to-end retrieval latency*) executada em amostra padronizada de 20 consultas do corpus técnico (CM Comandos).

## Especificações do Ambiente de Teste
- **CPU:** Intel(R) Core(TM) Ultra 7 265H (16 núcleos / 16 threads)
- **GPU:** NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU (8.151 MiB VRAM)
- **Modo CPU PyTorch (HuggingFace):** `CUDA_VISIBLE_DEVICES=""`, `device="cpu"`
- **Modo CPU Ollama (llama.cpp):** `options={"num_gpu": 0}`
- **Protocolo:** 1 consulta de *warmup* prévia (descartada) + 20 consultas cronometradas via `time.perf_counter()`

## Tabela Comparativa de Latência (Top-10 Retrieval)

| Modelo | Provedor | Dim. | Quantização | GPU Latência (ms) | CPU Média (ms) | CPU Desv. Pad. (ms) | CPU Min / Max (ms) | Desaceleração CPU (x) | Speedup GPU (x) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `multilingual_e5_large` | huggingface | 1024 | FP32 | 22.88 | 2850.85 | 606.70 | 2108.2 / 4221.1 | **124.61x** | **124.61x** |
| `bertimbau_sts` | huggingface | 768 | FP32 | 35.67 | 1439.92 | 520.58 | 878.0 / 2713.8 | **40.37x** | **40.37x** |
| `qwen3_4b_q4km` | ollama | 2560 | Q4_K_M | 35.98 | 870.72 | 380.34 | 351.6 / 1750.5 | **24.20x** | **24.20x** |
| `nomic_embed_text` | ollama | 768 | FP16/FP32 | 9.78 | 50.50 | 15.18 | 34.7 / 92.2 | **5.16x** | **5.16x** |

## Análise e Observações Técnicas

1. **Impacto da Arquitetura e Quantização:**
   - O modelo `multilingual_e5_large` (560M parâmetros, FP32 em PyTorch) apresenta a maior dependência de aceleração GPU, evidenciando alto custo computacional na CPU devido ao tamanho do transformer XLM-RoBERTa.
   - O modelo `bertimbau_sts` (110M parâmetros, FP32 em PyTorch) apresenta latência intermediária em CPU, proporcional à sua arquitetura BERT-base.
   - O modelo `qwen3_4b_q4km` (4B parâmetros, 4 bits GGUF via llama.cpp no Ollama) demonstra a eficiência de rotinas otimizadas com instruções AVX-512/AMX de inferência quantizada em CPU, mantendo latência viável mesmo com escala de parâmetros expressivamente superior aos modelos BERT.
   - O modelo `nomic_embed_text` (137M parâmetros via Ollama) opera com latência extremamente reduzida em CPU (< 50 ms), sendo o mais adaptável para arquiteturas embarcadas ou sem placa aceleradora dedicada.

2. **Relevância para a Tese de Conclusão de Curso (TCC):**
   - Para cenários de borda (*edge computing*) ou servidores locais desprovidos de GPU dedicada, `nomic_embed_text` e `qwen3_4b_q4km` oferecem compromissos favoráveis entre consumo de recursos e tempo de resposta.
   - Para ambientes de produção com requisitos estritos de SLA (< 100 ms por consulta) e prioridade máxima na qualidade de recuperação (*Hit Rate* e *MRR*), `multilingual_e5_large` exige infraestrutura acelerada por GPU.
