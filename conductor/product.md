# Product Definition: CM Comandos RAG Benchmark

## Vision & Overview
O projeto é um framework científico de experimentação e benchmark para arquiteturas de *Retrieval-Augmented Generation* (RAG), especializado no acervo técnico e manuais de engenharia de no-breaks e sistemas de energia crítica da fabricante CM Comandos Lineares. Seu objetivo primordial é avaliar quantitativamente a acurácia de recuperação de informação e geração textual entre modelos proprietários de referência (OpenAI `text-embedding-3-small` / GPT-4o-mini) e modelos abertos locais executados via Ollama e HuggingFace — com ênfase na viabilidade de modelos quantizados em 4 bits (`Q4_K_M`) e modelos multilíngues. O sistema combina rigor estatístico (testes não-paramétricos de Wilcoxon e bootstrapping), casamento multicamada determinístico contra *ground truth* e eficiência operacional em GPUs com restrição de memória, viabilizando aplicações corporativas com soberania de dados e fundamentando pesquisas acadêmicas de TCC.

## Target Audience & Use Cases
- **Pesquisadores Acadêmicos / TCC**: Avaliação formal de hipóteses científicas ($H_0$ e $H_1$) sobre a viabilidade de quantização k-quants (`Q4_K_M`) em tarefas de recuperação densa (*Information Retrieval*).
- **Engenharia e Suporte Técnico CM Comandos**: Assistente de consulta interativo e confiável para manuais de nobreaks industriais, diagramas elétricos e rotinas de manutenção preventiva/corretiva.
- **Ambientes Corporativos Regulados**: Implantação de RAG com soberania total de dados (*on-premise*), custo zero de API externa e latência ultra-baixa em hardware comercial (GPU de 8 GB).

## Key Features & Capabilities
1. **Pipeline de Benchmark de Recuperação (*Retriever-only*)**:
   - Isolamento estrito de índices vetoriais em disco por modelo (`storage/index_<model_id>/`).
   - Ciclo de vida de VRAM sequencial com descarga forçada (`keep_alive=0`, `torch.cuda.empty_cache()`, `gc.collect()`).
2. **Motor de Avaliação em Três Camadas (*Matching Engine*)**:
   - Casamento estrito por substring, similaridade por conjuntos de tokens (RapidFuzz) e sobreposição lexical ROUGE-L contra *ground truth*.
   - Cálculo automático de $HitRate@K$, $Recall@K$, $MRR@K$ e $MAP@K$ para $K \in \{2, 5, 10\}$.
3. **Consolidação Estatística e Visualização Científica**:
   - Teste pareado de postos com sinais de Wilcoxon ($\alpha = 0.05$) e Bootstrap CI 95% ($B = 1.000$).
   - Geração automatizada de gráficos em alta resolução (300 DPI) para artigos e monografias.
4. **Motor de Consulta e Chat RAG em Produção**:
   - Suporte a workflows LlamaIndex e interface interativa multiprocesso.
