# Análise Comparativa de Modelos de Embedding para Recuperação de Informação em Documentos Técnicos de Nobreaks

---

## Metadados do Documento

- **Título**: Análise Comparativa de Modelos de Embedding para Recuperação de Informação em Documentos Técnicos de Nobreaks
- **Subtítulo**: Avaliação Empírica de Modelos Proprietários, Multilíngues e Abertos Quantizados em 4 bits (Q4_K_M) em Arquitetura RAG Especializada
- **Contexto**: Trabalho de Conclusão de Curso (TCC) em Engenharia / Ciência da Computação
- **Autor / Pesquisador**: João Vitor Rura
- **Corpus de Teste**: 16 Manuais e Especificações Técnicas de Nobreaks e Sistemas de Energia Crítica (CM Comandos Lineares)
- **Data da Avaliação**: `{{DATA_AVALIACAO}}` (Exemplo: Setembro de 2026)
- **Status do Relatório**: Template Estruturado para Preenchimento Pós-Benchmark
- **Ambiente Computacional**:
  - **GPU**: NVIDIA RTX PRO 1000 Laptop GPU (8.192 MiB GDDR6 VRAM)
  - **Driver / CUDA**: Driver 550.x / CUDA 12.4
  - **Sistema Operacional**: Linux x86_64
  - **Runtime & Frameworks**: Python 3.12 (`uv`), LlamaIndex v0.10+, PyTorch 2.3+, Ollama 0.3+
  - **Repositório**: `orca/workspaces/rag_eval/gorgonian`

---

## 1. Resumo Executivo

> *Instruções de Preenchimento: Esta seção deve ser sintetizada em 3 a 5 parágrafos após a consolidação final dos dados empíricos, destacando o modelo vencedor, a aderência à hipótese H1 e as implicações práticas de custo e privacidade.*

`{{RESUMO_EXECUTIVO_TEXTO}}`

### Síntese dos Destaques Preliminares

| Dimensão Avaliada | Modelo de Destaque | Resultado / Observação Principal |
| :--- | :--- | :--- |
| **Maior Eficácia Geral (MRR@5)** | `{{MODELO_TOP_MRR}}` | `{{VALOR_TOP_MRR}}` (vs. Baseline OpenAI: `{{VALOR_OPENAI_MRR}}`) |
| **Maior Cobertura (Hit Rate@5)** | `{{MODELO_TOP_HITRATE}}` | `{{VALOR_TOP_HITRATE}}` |
| **Maior Eficiência (Latência / Consulta)** | `{{MODELO_MAIS_RAPIDO}}` | `{{LATENCIA_MINIMA_MS}}` ms |
| **Retenção da Hipótese H1 (≥85%)** | `{{STATUS_HIPOTESE_H1}}` | `{{NUMERO_MODELOS_ATINGIRAM_H1}}` de 6 modelos locais atingiram o critério |
| **Viabilidade em 8 GB VRAM** | `{{VIABILIDADE_VRAM_RESUMO}}` | Quantização Q4_K_M viabilizou modelos de até 7.6B parâmetros sem estouro de memória |

---

## 2. Objetivos e Hipóteses de Pesquisa

### 2.1 Objetivos

1. **Objetivo Geral**: Avaliar comparativamente a eficácia e a viabilidade operacional de modelos de representação vetorial densa (embeddings) abertos e quantizados frente ao modelo proprietário comercial da OpenAI (`text-embedding-3-small`) para recuperação de informação técnica em língua portuguesa no domínio de nobreaks.
2. **Objetivos Específicos**:
   - Quantificar o impacto da quantização de 4 bits em k-quants (`Q4_K_M`) na preservação do espaço latente e ordenação top-$K$;
   - Avaliar a hipótese de compensação paramétrica: verificar se modelos maiores quantizados (7.6B Q4) superam modelos menores não quantizados (560M FP16) na retenção semântica de termos técnicos;
   - Medir a eficiência computacional prática em GPU de uso pessoal/workstation (8 GB VRAM), analisando o trade-off entre latência, consumo de memória e acurácia;
   - Determinar a viabilidade de implantação com soberania de dados e custo zero de inferência via API em ambientes corporativos industriais.

### 2.2 Hipóteses Formais

Para fundamentar o estudo com o devido rigor científico exigido em um TCC, as hipóteses foram formalizadas em termos de paridade operacional e significância estatística:

- **Hipótese Nula ($H_0$)**: Não há viabilidade de substituição sem perdas severas — os modelos abertos locais quantizados em 4 bits (`Q4_K_M`) falham em atingir o patamar mínimo de 85% do desempenho do baseline proprietário da OpenAI em pelo menos duas métricas clássicas de Information Retrieval (IR), ou não apresentam aderência satisfatória:
  $$\mu_{\text{Métrica}}(M_{\text{local, Q4}}) < 0.85 \times \mu_{\text{Métrica}}(M_{\text{OpenAI}})$$
- **Hipótese Alternativa ($H_1$)**: Pelo menos um modelo aberto local quantizado em `Q4_K_M` atinge ou supera **85% do desempenho do baseline OpenAI** (`text-embedding-3-small`) em **pelo menos duas métricas de IR** entre Hit Rate@K, MRR@K, Context Recall@K e MAP@K (para $K=5$):
  $$\mu_{\text{Métrica}}(M_{\text{local, Q4}}) \ge 0.85 \times \mu_{\text{Métrica}}(M_{\text{OpenAI}}) \quad \text{em } \ge 2 \text{ métricas de IR}$$

### 2.3 Critério de Decisão Estatística

- **Teste Não-Paramétrico de Postos com Sinais de Wilcoxon** (*Wilcoxon signed-rank test*): aplicado par a par entre os resultados por consulta ($N=128$) do candidato local e o baseline OpenAI, avaliando se a diferença de postos é estatisticamente significante ($\alpha = 0.05$).
- **Intervalos de Confiança (IC 95%) por Bootstrap**: reamostragem não-paramétrica ($B = 1.000$ iterações com reposição) para estimar a estabilidade da média amostral de cada métrica sem pressupor normalidade da distribuição.

---

## 3. Configuração Experimental

### 3.1 Modelos Avaliados

O benchmark abrange 7 modelos selecionados estrategicamente para cobrir diferentes paradigmas arquiteturais, contagens de parâmetros, abordagens linguísticas e técnicas de quantização:

| ID | Nome do Modelo | Provedor / Runtime | Parâmetros | Quantização | Dimensão Vetorial | Contexto Máx. | Papel Experimental |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **M1** | `text-embedding-3-small` | OpenAI Cloud API | Não divulgado (~300M) | FP16 (SaaS) | 1.536 | 8.191 tokens | **Baseline Proprietário**: Referência comercial padrão de mercado. |
| **M2** | `qwen3-embedding:8b` | Ollama (Local) | 7.6B | `Q4_K_M` | 4.096 | 40.960 tokens | **Candidato Principal**: Modelo denso de alta escala testando o *sweet spot* de quantização. |
| **M3** | `qwen3-embedding:4b` | Ollama (Local) | 4.0B | `Q4_K_M` | 2.560 | 40.960 tokens | **Escala Intermediária**: Análise de degradação ao reduzir parâmetros mantendo quantização 4-bit. |
| **M4** | `bge-m3` | Ollama (Local) | 560M | FP16 / GGUF | 1.024 | 8.192 tokens | **Referência Aberta Multilíngue**: Estado-da-arte multilíngue aberto pré-treinado com suporte nativo a PT-BR. |
| **M5** | `multilingual-e5-large` | HuggingFace / FastEmbed | 560M | FP16 | 1.024 | 512 tokens | **Histórico de Produção**: Embedder padrão amplamente adotado em sistemas RAG corporativos. |
| **M6** | `nomic-embed-text` | Ollama (Local) | 137M | `Q4_K_M` | 768 | 8.192 tokens | **Alta Eficiência / Ultraleve**: Avaliação de viabilidade em cenários de extrema restrição de recursos. |
| **M7** | `bertimbau-base-portuguese-sts` | HuggingFace | 110M | FP32 / FP16 | 768 | 512 tokens | **Especialista em PT-BR**: Modelo BERT pré-treinado no BrWaC e ajustado para Similaridade Semântica (STS). |

### 3.2 Corpus Documental e Pré-Processamento

- **Origem dos Dados**: 16 manuais de instalação, operação e manutenção de nobreaks industriais da fabricante brasileira **CM Comandos Lineares** (série corporativa, nobreaks trifásicos e monofásicos, módulos de paralelismo e baterias VRLA seladas).
- **Segmentação Textual (*Chunking*)**:
  - Mecanismo: `SentenceSplitter` da biblioteca LlamaIndex;
  - Tamanho de Bloco ($L$): **512 tokens**;
  - Sobreposição (*Overlap* $O$): **50 tokens** (~10%);
  - Total de Nós Gerados no Corpus: Fixado e idêntico para todos os modelos avaliados.
- **Isolamento de Índices em Disco**:
  - Cada modelo possui sua pasta de persistência isolada em `storage/index_{model_id}/`;
  - Garante que a indexação, os vetores e os IDs de nós sejam reconstruídos rigorosamente com o respectivo vetor do modelo, prevenindo contaminação cruzada.

### 3.3 Conjunto de Teste (*Ground Truth*) e Deduplicação

- **Base de Consultas Sintetizada**: 132 perguntas técnicas associadas a respostas de referência e trechos canônicos (*reference contexts*), geradas a partir de `testset_openai_4omini.jsonl`.
- **Deduplicação Determinística**:
  - Processada via tupla hash `(user_input.strip(), tuple(reference_contexts))`;
  - Total Bruto de Consultas: **132**;
  - Duplicatas Detectadas e Removidas: **4**;
  - **Tamanho Efetivo da Amostra ($N_{\text{efetivo}}$)**: **128 consultas independentes**.

### 3.4 Procedimento de Recuperação (*Retriever-Only Evaluation*)

- Para cada consulta $q \in Q$ ($|Q|=128$):
  1. O modelo de embedding gera a representação vetorial $v_q$;
  2. O índice realiza busca por vizinhos mais próximos via **Similaridade de Cosseno**;
  3. São recuperados os nós ranqueados para $K \in \{2, 5, 10\}$;
  4. Não há geração por LLM na etapa de avaliação, isolando estritamente a eficácia do componente de recuperação (*retriever-only*).

### 3.5 Mecanismo Híbrido de Avaliação de Correspondência (*Matching Engine*)

Um nó recuperado $n$ é considerado relevante para o contexto de referência $r$ ($\text{Rel}(n, r) = 1$) se satisfizer ao menos uma das seguintes condições em cascata lógica (OR):
1. **Contenção Estrita de Substring**: se $r$ está contido em $n.\text{text}$ ou $n.\text{text}$ está contido em $r$ (para trechos com $\ge 40$ caracteres após normalização de whitespace e caixa baixa);
2. **Similaridade Fuzzy de Conjuntos de Tokens**: se o `token_set_ratio` (RapidFuzz) for $\ge 85\%$ ou `partial_ratio` $\ge 80\%$;
3. **Sobreposição Lexical ROUGE-L**: se o $F_1\text{-Score}$ da maior subsequência comum for $\ge 0.50$.

### 3.6 Métricas de Recuperação Implementadas

- **Hit Rate no Top-K ($\text{HR}@K$)**: Proporção de consultas em que ao menos um nó relevante foi retornado entre os $K$ primeiros nós:
  $$\text{HR}@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \sum_{j=1}^K \text{Rel}(n_{q,j}, \mathcal{R}_q) > 0 \right)$$
- **Mean Reciprocal Rank ($\text{MRR}@K$)**: Média dos inversos da primeira posição de rank onde um nó relevante aparece:
  $$\text{MRR}@K = \frac{1}{|Q|} \sum_{q \in Q} \left( \frac{1}{\min \{ j \in [1, K] \mid \text{Rel}(n_{q,j}, \mathcal{R}_q) = 1 \}} \right)$$
- **Context Recall no Top-K ($\text{CR}@K$)**: Proporção média de contextos de referência do gabarito que foram cobertos no Top-$K$:
  $$\text{CR}@K = \frac{1}{|Q|} \sum_{q \in Q} \frac{\left| \{ r \in \mathcal{R}_q \mid \exists j \le K \text{ que satisfaz } r \} \right|}{|\mathcal{R}_q|}$$
- **Mean Average Precision ($\text{MAP}@K$)**: Média da precisão calculada em cada ponto de corte onde um documento relevante é recuperado:
  $$\text{MAP}@K = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\min(|\mathcal{R}_q|, K)} \sum_{j=1}^K \text{Precision}@j(q) \cdot \text{Rel}(n_{q,j}, \mathcal{R}_q)$$

### 3.7 Controle Experimental e Hardware

- **Gestão de VRAM na GPU**: Execução rigorosamente sequencial. Ao término da avaliação de cada modelo, o buffer de VRAM é liberado de forma forçada (`keep_alive=0` no Ollama, `torch.cuda.empty_cache()` e `gc.collect()`).
- **Fixação de Sementes**: Semente $42$ fixada para `random`, `numpy` e `torch.cuda` garantindo reprodutibilidade matemática exata.

---

## 4. Resultados Experimentais

> *Nota: Os dados a seguir são consolidados automaticamente pelo script `scripts/consolidate_metrics.py` a partir das execuções salvas em `results/raw_retrievals_*.json`.*

### 4.1 Tabela Consolidada de Métricas de Recuperação

`{{TABELA_CONSOLIDADA}}`

#### Estrutura Detalhada por Nível de Top-K

| Modelo | Dim. | Quant. | HR@2 | HR@5 | HR@10 | MRR@2 | MRR@5 | MRR@10 | Rec@2 | Rec@5 | Rec@10 | MAP@2 | MAP@5 | MAP@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `openai_3_small` | 1536 | FP16 | `{{OAI_HR2}}` | `{{OAI_HR5}}` | `{{OAI_HR10}}` | `{{OAI_MRR2}}` | `{{OAI_MRR5}}` | `{{OAI_MRR10}}` | `{{OAI_REC2}}` | `{{OAI_REC5}}` | `{{OAI_REC10}}` | `{{OAI_MAP2}}` | `{{OAI_MAP5}}` | `{{OAI_MAP10}}` |
| `qwen3_8b_q4km` | 4096 | Q4_K_M | `{{Q8_HR2}}` | `{{Q8_HR5}}` | `{{Q8_HR10}}` | `{{Q8_MRR2}}` | `{{Q8_MRR5}}` | `{{Q8_MRR10}}` | `{{Q8_REC2}}` | `{{Q8_REC5}}` | `{{Q8_REC10}}` | `{{Q8_MAP2}}` | `{{Q8_MAP5}}` | `{{Q8_MAP10}}` |
| `qwen3_4b_q4km` | 2560 | Q4_K_M | `{{Q4_HR2}}` | `{{Q4_HR5}}` | `{{Q4_HR10}}` | `{{Q4_MRR2}}` | `{{Q4_MRR5}}` | `{{Q4_MRR10}}` | `{{Q4_REC2}}` | `{{Q4_REC5}}` | `{{Q4_REC10}}` | `{{Q4_MAP2}}` | `{{Q4_MAP5}}` | `{{Q4_MAP10}}` |
| `bge_m3` | 1024 | FP16/GGUF | `{{BGE_HR2}}` | `{{BGE_HR5}}` | `{{BGE_HR10}}` | `{{BGE_MRR2}}` | `{{BGE_MRR5}}` | `{{BGE_MRR10}}` | `{{BGE_REC2}}` | `{{BGE_REC5}}` | `{{BGE_REC10}}` | `{{BGE_MAP2}}` | `{{BGE_MAP5}}` | `{{BGE_MAP10}}` |
| `multilingual_e5_large`| 1024 | FP16 | `{{E5_HR2}}` | `{{E5_HR5}}` | `{{E5_HR10}}` | `{{E5_MRR2}}` | `{{E5_MRR5}}` | `{{E5_MRR10}}` | `{{E5_REC2}}` | `{{E5_REC5}}` | `{{E5_REC10}}` | `{{E5_MAP2}}` | `{{E5_MAP5}}` | `{{E5_MAP10}}` |
| `nomic_embed_text` | 768 | Q4_K_M | `{{NOM_HR2}}` | `{{NOM_HR5}}` | `{{NOM_HR10}}` | `{{NOM_MRR2}}` | `{{NOM_MRR5}}` | `{{NOM_MRR10}}` | `{{NOM_REC2}}` | `{{NOM_REC5}}` | `{{NOM_REC10}}` | `{{NOM_MAP2}}` | `{{NOM_MAP5}}` | `{{NOM_MAP10}}` |
| `bertimbau_sts` | 768 | FP16 | `{{BER_HR2}}` | `{{BER_HR5}}` | `{{BER_HR10}}` | `{{BER_MRR2}}` | `{{BER_MRR5}}` | `{{BER_MRR10}}` | `{{BER_REC2}}` | `{{BER_REC5}}` | `{{BER_REC10}}` | `{{BER_MAP2}}` | `{{BER_MAP5}}` | `{{BER_MAP10}}` |

---

### 4.2 Testes de Significância Estatística (Wilcoxon Signed-Rank Test)

Comparação pareada de cada modelo local candidato contra o baseline OpenAI (`text-embedding-3-small`) para $N=128$ pares de teste no ponto de corte $K=5$:

`{{TABELA_WILCOXON}}`

#### Gabarito de Decisão do Teste Pareado

| Modelo Local Candidato | Estatística $W$ (MRR@5) | p-valor (MRR@5) | Estatística $W$ (HR@5) | p-valor (HR@5) | Conclusão Estatística ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `qwen3_8b_q4km` | `{{W_MRR_Q8}}` | `{{P_MRR_Q8}}` | `{{W_HR_Q8}}` | `{{P_HR_Q8}}` | `{{CONCLUSAO_ESTATISTICA_Q8}}` |
| `qwen3_4b_q4km` | `{{W_MRR_Q4}}` | `{{P_MRR_Q4}}` | `{{W_HR_Q4}}` | `{{P_HR_Q4}}` | `{{CONCLUSAO_ESTATISTICA_Q4}}` |
| `bge_m3` | `{{W_MRR_BGE}}` | `{{P_MRR_BGE}}` | `{{W_HR_BGE}}` | `{{P_HR_BGE}}` | `{{CONCLUSAO_ESTATISTICA_BGE}}` |
| `multilingual_e5_large`| `{{W_MRR_E5}}` | `{{P_MRR_E5}}` | `{{W_HR_E5}}` | `{{P_HR_E5}}` | `{{CONCLUSAO_ESTATISTICA_E5}}` |
| `nomic_embed_text` | `{{W_MRR_NOM}}` | `{{P_MRR_NOM}}` | `{{W_HR_NOM}}` | `{{P_HR_NOM}}` | `{{CONCLUSAO_ESTATISTICA_NOM}}` |
| `bertimbau_sts` | `{{W_MRR_BER}}` | `{{P_MRR_BER}}` | `{{W_HR_BER}}` | `{{P_HR_BER}}` | `{{CONCLUSAO_ESTATISTICA_BER}}` |

*Critério: Se $p \ge 0.05$, a hipótese de diferença significante é rejeitada (não há evidência estatística de que o modelo local performa diferente do baseline da OpenAI).*

---

### 4.3 Intervalos de Confiança via Bootstrap (IC 95%)

`{{TABELA_BOOTSTRAP_CI}}`

#### Intervalos Estimados com $B = 1.000$ Reamostragens

| Modelo | Hit Rate@5 [IC 95%] | MRR@5 [IC 95%] | Context Recall@5 [IC 95%] | MAP@5 [IC 95%] |
| :--- | :---: | :---: | :---: | :---: |
| `openai_3_small` (Base) | `{{CI_OAI_HR5}}` | `{{CI_OAI_MRR5}}` | `{{CI_OAI_REC5}}` | `{{CI_OAI_MAP5}}` |
| `qwen3_8b_q4km` | `{{CI_Q8_HR5}}` | `{{CI_Q8_MRR5}}` | `{{CI_Q8_REC5}}` | `{{CI_Q8_MAP5}}` |
| `qwen3_4b_q4km` | `{{CI_Q4_HR5}}` | `{{CI_Q4_MRR5}}` | `{{CI_Q4_REC5}}` | `{{CI_Q4_MAP5}}` |
| `bge_m3` | `{{CI_BGE_HR5}}` | `{{CI_BGE_MRR5}}` | `{{CI_BGE_REC5}}` | `{{CI_BGE_MAP5}}` |
| `multilingual_e5_large` | `{{CI_E5_HR5}}` | `{{CI_E5_MRR5}}` | `{{CI_E5_REC5}}` | `{{CI_E5_MAP5}}` |
| `nomic_embed_text` | `{{CI_NOM_HR5}}` | `{{CI_NOM_MRR5}}` | `{{CI_NOM_REC5}}` | `{{CI_NOM_MAP5}}` |
| `bertimbau_sts` | `{{CI_BER_HR5}}` | `{{CI_BER_MRR5}}` | `{{CI_BER_REC5}}` | `{{CI_BER_MAP5}}` |

---

### 4.4 Eficiência Computacional: Latência e Tempo de Indexação

`{{TABELA_LATENCIA_INDEXACAO}}`

#### Desempenho Operacional em Ambiente Local

| Modelo | Tempo de Indexação (s) | Latência Média de Consulta (ms) | Consultas por Segundo (QPS) | Tamanho do Índice em Disco (MB) |
| :--- | :---: | :---: | :---: | :---: |
| `openai_3_small` | `{{BUILD_TIME_OAI}}` | `{{LAT_OAI}}` | `{{QPS_OAI}}` | `{{SIZE_OAI}}` |
| `qwen3_8b_q4km` | `{{BUILD_TIME_Q8}}` | `{{LAT_Q8}}` | `{{QPS_Q8}}` | `{{SIZE_Q8}}` |
| `qwen3_4b_q4km` | `{{BUILD_TIME_Q4}}` | `{{LAT_Q4}}` | `{{QPS_Q4}}` | `{{SIZE_Q4}}` |
| `bge_m3` | `{{BUILD_TIME_BGE}}` | `{{LAT_BGE}}` | `{{QPS_BGE}}` | `{{SIZE_BGE}}` |
| `multilingual_e5_large` | `{{BUILD_TIME_E5}}` | `{{LAT_E5}}` | `{{QPS_E5}}` | `{{SIZE_E5}}` |
| `nomic_embed_text` | `{{BUILD_TIME_NOM}}` | `{{LAT_NOM}}` | `{{QPS_NOM}}` | `{{SIZE_NOM}}` |
| `bertimbau_sts` | `{{BUILD_TIME_BER}}` | `{{LAT_BER}}` | `{{QPS_BER}}` | `{{SIZE_BER}}` |

---

### 4.5 Visualizações Gráficas

Os gráficos correspondentes são gerados automaticamente pelo módulo `scripts/plot_embedding_graphs.py` e armazenados no diretório `graficos_tcc/`:

1. **Ranking por MRR@5 e Hit Rate@5**:
   `{{GRAFICO_RANKING_K5}}`
   *(Localização: `graficos_tcc/ranking_hitrate_mrr_k5.png`)*
   
2. **Curva de Evolução do Recall por Top-K ($K \in \{2, 5, 10\}$)**:
   `{{GRAFICO_CURVA_TOPK}}`
   *(Localização: `graficos_tcc/curva_recuperacao_topk.png`)*

3. **Trade-off de Engenharia: Eficiência (Latência em ms) vs. Eficácia (MRR@5)**:
   `{{GRAFICO_TRADEOFF_LATENCIA}}`
   *(Localização: `graficos_tcc/tradeoff_latencia_mrr.png`)*

---

## 5. Análise da Hipótese H1 e Discussão Científica

### 5.1 Matriz de Validação da Hipótese H1

A validação de $H_1$ exige que a razão relativa entre a métrica do modelo aberto e o baseline OpenAI seja $\ge 0.85$ (retenção $\ge 85\%$) em pelo menos duas métricas de IR no ponto de corte canônico $K=5$:

$$\text{Retenção}(\%) = \left( \frac{\text{Métrica}(M_{\text{candidato}})}{\text{Métrica}(M_{\text{OpenAI}})} \right) \times 100\%$$

| Modelo Candidato | % Retenção HR@5 | % Retenção MRR@5 | % Retenção Rec@5 | % Retenção MAP@5 | Critério Atendido (≥ 2 métricas ≥ 85%)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `qwen3_8b_q4km` | `{{RET_HR_Q8}}%` | `{{RET_MRR_Q8}}%` | `{{RET_REC_Q8}}%` | `{{RET_MAP_Q8}}%` | `{{STATUS_H1_Q8}}` |
| `qwen3_4b_q4km` | `{{RET_HR_Q4}}%` | `{{RET_MRR_Q4}}%` | `{{RET_REC_Q4}}%` | `{{RET_MAP_Q4}}%` | `{{STATUS_H1_Q4}}` |
| `bge_m3` | `{{RET_HR_BGE}}%` | `{{RET_MRR_BGE}}%` | `{{RET_REC_BGE}}%` | `{{RET_MAP_BGE}}%` | `{{STATUS_H1_BGE}}` |
| `multilingual_e5_large`| `{{RET_HR_E5}}%` | `{{RET_MRR_E5}}%` | `{{RET_REC_E5}}%` | `{{RET_MAP_E5}}%` | `{{STATUS_H1_E5}}` |
| `nomic_embed_text` | `{{RET_HR_NOM}}%` | `{{RET_MRR_NOM}}%` | `{{RET_REC_NOM}}%` | `{{RET_MAP_NOM}}%` | `{{STATUS_H1_NOM}}` |
| `bertimbau_sts` | `{{RET_HR_BER}}%` | `{{RET_MRR_BER}}%` | `{{RET_REC_BER}}%` | `{{RET_MAP_BER}}%` | `{{STATUS_H1_BER}}` |

### 5.2 Discussão Detalhada dos Resultados

`{{DISCUSSAO_HIPOTESE_H1}}`

#### Tópicos de Análise a Explorar:
1. **O Efeito do Sweet Spot `Q4_K_M`**:
   - Os resultados corroboram que a quantização em 4 bits preserva a geometria esférica do espaço de cosseno?
   - Houve distorção na ordenação de vizinhos mais próximos no Top-2 vs Top-5?
2. **Escala de Parâmetros vs Precisão Numérica**:
   - O modelo `Qwen3-8B` quantizado em 4 bits superou modelos em precisão completa FP16 de porte menor (`BGE-M3` e `E5-Large` de 560M)?
   - Em que ponto a redução de parâmetros para 4B ou 137M degrada a capacidade de capturar terminologias elétricas de nobreaks?
3. **Especialização de Domínio e Idioma**:
   - Como o `bertimbau-base-portuguese-sts` performou em comparação com os modelos multilíngues mais modernos? A especialização sintática no português do Brasil compensa a defasagem temporal de arquitetura (BERT vs Llama/Qwen)?

---

## 6. Análise de Viabilidade Prática e Engenharia

### 6.1 Consumo de Memória (VRAM) e Alocação na GPU

A infraestrutura de teste utilizou uma GPU **NVIDIA RTX PRO 1000 com 8 GB de VRAM**, uma placa de nível intermediário/profissional encontrada em estações móveis de engenharia:

| Modelo | Pegada em Disco (Pesos) | VRAM Alocada na GPU | Margem Livre em 8 GB | Viabilidade em GPU 8 GB |
| :--- | :---: | :---: | :---: | :---: |
| `text-embedding-3-small` | 0 MB (Nuvem) | 0 MB | 8.192 MB (100%) | Perfeita (Sem GPU Local) |
| `qwen3-embedding:8b` (Q4_K_M) | ~4.7 GB | ~5.2 GB | ~2.9 GB (~36%) | **Viável com folga** |
| `qwen3-embedding:4b` (Q4_K_M) | ~2.5 GB | ~2.9 GB | ~5.2 GB (~65%) | **Muito Viável** |
| `bge-m3` (FP16 / GGUF) | ~2.2 GB | ~2.6 GB | ~5.5 GB (~68%) | **Muito Viável** |
| `multilingual-e5-large` (FP16) | ~2.2 GB | ~2.5 GB | ~5.6 GB (~70%) | **Muito Viável** |
| `nomic-embed-text` (Q4_K_M) | ~274 MB | ~450 MB | ~7.7 GB (~94%) | **Excelente (Ultra Leve)** |
| `bertimbau-base-portuguese-sts` | ~435 MB | ~650 MB | ~7.5 GB (~92%) | **Excelente** |

> **Achado de Engenharia**: O modelo `qwen3-embedding:8b` em formato `Q4_K_M` consumiu aproximadamente 5.2 GB de VRAM durante a inferência com batch unitário, permitindo que a aplicação RAG completa (incluindo vetor store local ou pequeno SLM gerador) opere integralmente dentro dos limites de uma GPU de 8 GB sem transbordamento para a memória RAM do sistema (*paging/swapping*).

### 6.2 Latência e Experiência do Usuário em Tempo Real

- Limite aceitável de latência para a etapa de recuperação em um chat RAG interativo: **≤ 150 ms**;
- Comportamento observado nos testes:
  `{{ANALISE_LATENCIA_PRODUCAO}}`
- Impacto do overhead de rede no modelo OpenAI vs inferência direta via barramento PCIe local.

### 6.3 Análise Econômica (Custo de API vs TCO Local)

- **Custo OpenAI (`text-embedding-3-small`)**: US$ 0,02 por milhão de tokens.
  - Para um corpus estático de 16 manuais (aproximadamente 300.000 tokens), o custo de indexação é desprezível (< US$ 0,01);
  - Porém, em regime contínuo de atendimento corporativo (milhares de consultas diárias, reindexação periódica de versões e documentações operacionais de centenas de produtos), o custo cumulativo e a dependência financeira tornam-se fatores recorrentes.
- **Modelos Locais (`Qwen3`, `BGE-M3`)**: **Custo zero de API**.
  - Amortização total sobre a infraestrutura de hardware já existente (servidor interno ou workstation do setor de engenharia).

### 6.4 Soberania, Privacidade e Segurança da Informação

- Documentos técnicos de equipamentos de energia crítica (nobreaks trifásicos, diagramas de placas de controle PWM, procedimentos de bypass estático) frequentemente contêm **segredos industriais**, esquemáticos proprietários e dados de clientes sob acordos de confidencialidade (NDA).
- O envio de fragmentos textuais para a API externa da OpenAI viola políticas rígidas de segurança corporativa em setores regulados (bancário, militar, hospitalar, datacenters).
- A execução local (via Ollama ou PyTorch) garante **soberania de dados absoluta**, mantendo 100% dos dados dentro do perímetro da rede corporativa da empresa.

---

## 7. Ameaças à Validade

### 7.1 Validade Interna: O Viés de Circularidade do Ground Truth

- **Natureza da Ameaça**: O conjunto de teste de referência (`testset_openai_4omini.jsonl`) foi sintetizado com auxílio do modelo de linguagem GPT-4o-mini a partir de índices construídos com o `text-embedding-3-small` da OpenAI.
- **Risco**: Essa arquitetura de geração introduz uma **afinidade semântica circular intrínseca** em favor do baseline M1 (`openai_3_small`), visto que o GPT-4o-mini tende a formular perguntas cujas chaves lexicais e de estilo espelham a própria projeção dos embeddings da OpenAI.
- **Estratégia de Mitigação Implementada**:
  1. Utilização de um **mecanismo de casamento em três camadas** (`matching_engine.py`) combinando contenção estrita, fuzzy matching por conjuntos de tokens (RapidFuzz $\ge 85\%$) e sobreposição lexical ROUGE-L ($F_1 \ge 0.50$), evitando penalizar sinônimos técnicos e pequenas diferenças de partição;
  2. **Auditoria Qualitativa Manual de 20 Amostras**: Seleção aleatória de 20 consultas do dataset de teste para validação manual cega, investigando casos onde modelos abertos recuperaram nós tecnicamente pertinentes que divergiram formalmente do gabarito sintético:

`{{AUDITORIA_QUALITATIVA_AMOSTRAS}}`

#### Estrutura do Relatório de Auditoria das 20 Amostras

| ID Consulta | Pergunta Técnica | Resposta Esperada | Nó Recuperado (Modelo Aberto) | Avaliação Humana | Veredito |
| :---: | :--- | :--- | :--- | :--- | :---: |
| #01 | `{{PERGUNTA_01}}` | `{{GABARITO_01}}` | `{{RECUPERADO_01}}` | `{{AVAL_01}}` | `{{VEREDITO_01}}` |
| #02 | `{{PERGUNTA_02}}` | `{{GABARITO_02}}` | `{{RECUPERADO_02}}` | `{{AVAL_02}}` | `{{VEREDITO_02}}` |
| ... | ... | ... | ... | ... | ... |
| #20 | `{{PERGUNTA_20}}` | `{{GABARITO_20}}` | `{{RECUPERADO_20}}` | `{{AVAL_20}}` | `{{VEREDITO_20}}` |

### 7.2 Validade de Construto: Granularidade de Fragmentação (*Chunking*)

- O uso do `SentenceSplitter` com tamanho de 512 tokens e overlap de 50 tokens pode fragmentar tabelas de especificações técnicas elétricas (como curvas de rendimento, faixas de tensão de entrada/saída e capacidades de baterias).
- Como todos os 7 modelos foram submetidos exatamente à mesma divisão de nós em memória e em disco, a consistência comparativa foi mantida estritamente uniforme.

### 7.3 Validade Externa: Generalização do Domínio

- As conclusões empíricas deste benchmark são diretamente válidas para o domínio de **engenharia elétrica, automação e equipamentos de energia ininterrupta (nobreaks)** com documentação em português do Brasil.
- A generalização para corpora de domínio aberto ou textos puramente narrativos/literários não pode ser inferida sem experimentação complementar.

---

## 8. Conclusões e Recomendações

### 8.1 Conclusão Geral

`{{CONCLUSAO_GERAL}}`

### 8.2 Recomendações de Engenharia para o Sistema RAG

`{{RECOMENDACOES_ENGENHARIA}}`

1. **Modelo Recomendado para Produção**: `{{MODELO_RECOMENDADO_FINAL}}`;
2. **Justificativa do Trade-off**: Balanço entre cobertura no Top-5 ($HR@5$), precisão de primeiro posto ($MRR@5$), latência em GPU e custo zero de infraestrutura;
3. **Estratégia de Deploy**: Implantação empacotada via container Docker utilizando o serviço Ollama para abstração de drivers de GPU e facilidade de substituição de modelos sem alteração no código da aplicação LlamaIndex.

### 8.3 Propostas de Trabalhos Futuros

`{{TRABALHOS_FUTUROS}}`

- Investigar a combinação de recuperação densa com **recuperação esparsa híbrida (BM25 + Splade / BGE-M3 Sparse)**;
- Avaliar a inserção de uma camada de **Re-ranking neural** (Cross-Encoder como `bge-reranker-large` quantizado em 4 bits) sobre o Top-10 retornado pelo embedder denso;
- Explorar o ajuste fino de representação (*contrastive fine-tuning*) via LoRA nos manuais da CM Comandos para alinhar jargões exclusivos e nomenclaturas fabris da empresa.

---

## 9. Referências Bibliográficas

1. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.** (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems (NeurIPS 2023), 36. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
2. **Efron, B., & Tibshirani, R. J.** (1994). *An Introduction to the Bootstrap*. CRC press / Chapman & Hall.
3. **Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D.** (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. International Conference on Learning Representations (ICLR 2023). [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).
4. **Kawrakow, G.** (2023). *k-quants: Fast and accurate 4-bit, 5-bit, and 6-bit quantization in llama.cpp*. Repositório oficial llama.cpp, pull request #1684. [Disponível em GitHub](https://github.com/ggerganov/llama.cpp/pull/1684).
5. **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D.** (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 9459-9474.
6. **Lin, J., Nogueira, R., & Guo, C.** (2021). *Pretrained Transformers for Text Ranking: BERT and Beyond*. Synthesis Lectures on Human Language Technologies, Morgan & Claypool Publishers.
7. **Liu, J.** (2023). *LlamaIndex: Data framework for LLM-based applications*. [Disponível em LlamaIndex](https://www.llamaindex.ai).
8. **Neelakantan, A., Xu, T., Puri, R., Radford, A., Han, J. M., Tworek, J., ... & Weng, L.** (2022). *Text and Code Embeddings by Contrastive Pre-Training Using Large Models*. OpenAI Research. [arXiv:2201.10005](https://arxiv.org/abs/2201.10005).
9. **Nussbaum, Z., Morris, J. X., Duderstadt, B., & Moshkovitz, M.** (2024). *Nomic Embed: Training a Reproducible Long Context Text Embedder*. Nomic AI. [arXiv:2402.01613](https://arxiv.org/abs/2402.01613).
10. **Souza, F., Nogueira, R., & Lotufo, R.** (2020). *BERTimbau: Pretrained BERT Models for Brazilian Portuguese*. 9th Brazilian Conference on Intelligent Systems (BRACIS 2020), Lecture Notes in Computer Science, vol 12319. Springer, Cham. [arXiv:2009.00658](https://arxiv.org/abs/2009.00658).
11. **Wang, L., Yang, N., Huang, F., Jiao, B., Yang, L., Jiang, D., ... & Wei, F.** (2022). *Text Embeddings by Weakly-Supervised Pre-Training (E5)*. Microsoft Research. [arXiv:2212.03533](https://arxiv.org/abs/2212.03533).
12. **Wilcoxon, F.** (1945). *Individual comparisons by ranking methods*. Biometrics Bulletin, 1(6), 80-83.
13. **Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N.** (2023). *C-Pack: Packaged Resources to Advance General Chinese and Multilingual Embedding Technologies (BGE-M3)*. Beijing Academy of Artificial Intelligence (BAAI). [arXiv:2309.07597](https://arxiv.org/abs/2309.07597).
