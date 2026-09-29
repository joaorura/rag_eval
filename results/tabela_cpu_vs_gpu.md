# Comparativo de Latência de Inferência: CPU vs GPU (Medição Corrigida e Padronizada)

Avaliação empírica do tempo de recuperação por consulta (*end-to-end retrieval latency*) sob condições idênticas de isolamento no corpus técnico da CM Comandos Lineares.

## Especificações do Hardware
- **CPU:** Intel(R) Core(TM) Ultra 7 265H (16 núcleos / 16 threads)
- **GPU:** NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU (8.151 MiB GDDR6 VRAM)
- **Runtime CPU Ollama:** llama.cpp com instruções AVX-512 e `options={"num_gpu": 0}`
- **Runtime CPU ONNX:** ONNX Runtime 1.30 com instruções `AVX-512 VNNI` quantizado em `INT8`

## Tabela Comparativa de Latência (Top-10 Retrieval)

| Modelo | Runtime / Quantização | Dimensão | Latência GPU | Latência CPU (Média) | Speedup GPU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`nomic_embed_text`** | Ollama GGUF (Q4_K_M) | 768 | **9,78 ms** | **50,50 ms** | **5,2×** |
| **`multilingual_e5_base`** | ONNX INT8 (AVX-512 VNNI) | 768 | ~12 ms | **53,34 ms** | **4,4×** |
| **`multilingual_e5_large`** | ONNX INT8 (AVX-512 VNNI) | 1024 | ~20 ms | **107,47 ms** | **5,4×** |
| **`qwen3_4b_q4km`** | Ollama GGUF (Q4_K_M) | 2560 | **35,98 ms** | **208,17 ms** | **5,8×** |
| **`qwen3_8b_q4km`** | Ollama GGUF (Q4_K_M) | 4096 | **57,06 ms** | **379,43 ms** | **6,6×** |
| **`openai_3_small`** *(Ref.)* | SaaS Cloud API | 1536 | **415,07 ms** | — | — |

*Nota: Em medição de inferência isolada de embedding na CPU (apenas forward pass do vetor), o `qwen3_4b_q4km` atinge 134,34 ms e o `qwen3_8b_q4km` atinge 294,65 ms, respeitando estritamente a lei de proporcionalidade pelo número de parâmetros.*
