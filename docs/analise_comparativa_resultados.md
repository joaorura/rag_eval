# Análise Comparativa de Modelos de Embedding para Recuperação de Informação em Documentos Técnicos de Nobreaks

---

## Metadados do Documento

- **Título**: Análise Comparativa de Modelos de Embedding para Recuperação de Informação em Documentos Técnicos de Nobreaks
- **Subtítulo**: Avaliação Empírica de Modelos Proprietários, Multilíngues e Abertos Quantizados em 4 bits (Q4_K_M) em Arquitetura RAG Especializada
- **Contexto**: Projeto de Pesquisa Científica e Tecnológica (ICT) em Engenharia / Ciência da Computação
- **Autor / Pesquisador**: João Messias Lima Pereira
- **Corpus de Teste**: 16 Manuais e Especificações Técnicas de Nobreaks e Sistemas de Energia Crítica (CM Comandos Lineares)
- **Data da Avaliação**: `29 de Setembro de 2026` (Exemplo: Setembro de 2026)
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

`Este relatório apresenta os resultados empíricos da avaliação comparativa de 7 modelos de representação vetorial densa (*embeddings*) sobre o corpus técnico de 16 manuais de nobreaks industriais da fabricante brasileira CM Comandos Lineares. O estudo compõe a etapa de validação da camada de recuperação (*Retriever-only*) do Projeto de Pesquisa Científica e Tecnológica (ICT), investigando a viabilidade de substituir APIs proprietárias em nuvem (OpenAI `text-embedding-3-small`) por modelos abertos e locais quantizados em 4 bits (`Q4_K_M`), garantindo custo zero de inferência, baixa latência e soberania absoluta sobre dados industriais sensíveis.

A avaliação foi conduzida sob 128 consultas técnicas deduplicadas em um arranjo experimental rigorosamente padronizado (*SentenceSplitter* de 512 tokens com sobreposição de 50 tokens; execução sequencial em GPU NVIDIA RTX PRO 1000 com 8 GB VRAM). As métricas de *Information Retrieval* avaliadas abrangeram Hit Rate@K, Mean Reciprocal Rank (MRR@K), Context Recall@K e Mean Average Precision (MAP@K) para $K \in \{2, 5, 10\}$, complementadas por testes pareados de Wilcoxon e intervalos de confiança via Bootstrap com $B=1.000$ iterações.

Os resultados empíricos confirmaram integralmente a **Hipótese Alternativa ($H_1$)**: os modelos quantizados locais da família Qwen3 (`qwen3_8b_q4km` e `qwen3_4b_q4km`) não apenas atingiram a meta mínima de 85% de retenção de desempenho em relação ao baseline da OpenAI, como o **superaram em três das quatro métricas de recuperação** no ponto de corte $K=5$. O modelo `qwen3_8b_q4km` atingiu Hit Rate@5 de 0,5703 (retenção de 121,65% vs. 0,4688 do baseline OpenAI) e MRR@5 de 0,3887 (retenção de 101,22%), mantendo equivalência estatística no teste pareado de Wilcoxon ($p = 0,9336$).

Em termos de eficiência de engenharia, a inferência local com k-quants demonstrou superioridade marcante: o modelo `qwen3_4b_q4km` operou com latência média de **35,98 ms por consulta** (sendo **11,5 vezes mais rápido** que o baseline da OpenAI, cuja latência média em nuvem atingiu 415,07 ms), consumindo apenas ~2,9 GB de VRAM. No topo geral de precisão de ranqueamento, o modelo aberto `multilingual-e5-large` alcançou o maior MRR@5 da bancada (0,4031 com latência de 22,88 ms). Por outro lado, modelos monolíngues sem pré-treinamento contrastivo massivo (`bertimbau_sts`) e modelos ultraleves (`nomic_embed_text`) apresentaram quedas estatisticamente significativas de desempenho ($p < 0,05$), evidenciando que a escala paramétrica compensa com folga a perda por quantização.`

### Síntese dos Destaques Preliminares

| Dimensão Avaliada | Modelo de Destaque | Resultado / Observação Principal |
| :--- | :--- | :--- |
| **Maior Eficácia Geral (MRR@5)** | `multilingual_e5_large` | `0,4031` (vs. Baseline OpenAI: `0,3840`) |
| **Maior Cobertura (Hit Rate@5)** | `qwen3_8b_q4km` | `0,5703 (57,03%)` |
| **Maior Eficiência (Latência / Consulta)** | `nomic_embed_text` | `9,78` ms |
| **Retenção da Hipótese H1 (≥85%)** | `Confirmada Plenamente (4 de 4 métricas ≥ 85%)` | `4` de 6 modelos locais atingiram o critério |
| **Viabilidade em 8 GB VRAM** | `100% Viável (pico de alocação de ~5,2 GB em 8 GB VRAM)` | Quantização Q4_K_M viabilizou modelos de até 7.6B parâmetros sem estouro de memória |

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

Para fundamentar o estudo com o devido rigor científico exigido no projeto de pesquisa aplicada (ICT), as hipóteses foram formalizadas em termos de paridade operacional e significância estatística:

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

`| Modelo | Quantização | Dim. | HR@5 | MRR@5 | Recall@5 | MAP@5 | Latência (ms) | Wilcoxon p (MRR@5) | Wilcoxon p (HR@5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `multilingual_e5_large` | FP16/FP32 | 1024 | **0.5469** | **0.4031** | **0.5110** | 0.6523 | 22.88 | 0.5217 | 0.0124* |
| `qwen3_8b_q4km` | Q4_K_M | 4096 | **0.5703** | 0.3887 | **0.5311** | 0.6836 | 57.06 | 0.9336 | 0.0029* |
| `qwen3_4b_q4km` | Q4_K_M | 2560 | 0.5547 | 0.3844 | 0.4965 | 0.6610 | 35.98 | 0.9565 | 0.0343* |
| `openai_3_small` (Base) | FP16/FP32 | 1536 | 0.4688 | 0.3840 | 0.4482 | 0.7286 | 415.07 | — | — |
| `bge_m3` | FP16/GGUF | 1024 | 0.5234 | 0.3797 | 0.4673 | **0.7441** | 167.30 | 0.8273 | 0.1266 |
| `nomic_embed_text` | FP16/GGUF | 768 | 0.3594 | 0.3021 | 0.3466 | 0.5545 | **9.78** | 0.0142* | 0.0017* |
| `bertimbau_sts` | FP16/FP32 | 768 | 0.4531 | 0.2423 | 0.3498 | 0.3620 | 35.67 | 0.0008* | 0.7456 |`

#### Estrutura Detalhada por Nível de Top-K

| Modelo | Dim. | Quant. | HR@2 | HR@5 | HR@10 | MRR@2 | MRR@5 | MRR@10 | Rec@2 | Rec@5 | Rec@10 | MAP@2 | MAP@5 | MAP@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `openai_3_small` | 1536 | FP16 | `0.3828` | `0.4688` | `0.5156` | `0.3594` | `0.3840` | `0.3898` | `0.3776` | `0.4482` | `0.4797` | `0.4473` | `0.7286` | `0.9428` |
| `qwen3_8b_q4km` | 4096 | Q4_K_M | `0.3906` | `0.5703` | `0.6172` | `0.3398` | `0.3887` | `0.3944` | `0.3688` | `0.5311` | `0.5593` | `0.3984` | `0.6836` | `0.9503` |
| `qwen3_4b_q4km` | 2560 | Q4_K_M | `0.3828` | `0.5547` | `0.6328` | `0.3398` | `0.3844` | `0.3949` | `0.3698` | `0.4965` | `0.5584` | `0.3984` | `0.6610` | `0.9234` |
| `bge_m3` | 1024 | FP16/GGUF | `0.3906` | `0.5234` | `0.6875` | `0.3438` | `0.3797` | `0.4020` | `0.3699` | `0.4673` | `0.5871` | `0.4355` | `0.7441` | `0.9945` |
| `multilingual_e5_large`| 1024 | FP16 | `0.3984` | `0.5469` | `0.6797` | `0.3672` | `0.4031` | `0.4204` | `0.3779` | `0.5110` | `0.6005` | `0.4336` | `0.6523` | `0.8908` |
| `nomic_embed_text` | 768 | Q4_K_M | `0.3203` | `0.3594` | `0.4531` | `0.2891` | `0.3021` | `0.3143` | `0.3151` | `0.3466` | `0.4253` | `0.3457` | `0.5545` | `0.7620` |
| `bertimbau_sts` | 768 | FP16 | `0.2188` | `0.4531` | `0.5391` | `0.1797` | `0.2423` | `0.2542` | `0.1936` | `0.3498` | `0.4100` | `0.1895` | `0.3620` | `0.5472` |

---

### 4.2 Testes de Significância Estatística (Wilcoxon Signed-Rank Test)

Comparação pareada de cada modelo local candidato contra o baseline OpenAI (`text-embedding-3-small`) para $N=128$ pares de teste no ponto de corte $K=5$:

`O teste pareado de Wilcoxon (*two-sided signed-rank test*) foi computado para cada consulta pareada ($N=128$) contra o baseline OpenAI `text-embedding-3-small` no ponto de corte $K=5$, avaliando as hipóteses nulas de equivalência em ordenação (MRR@5) e presença (Hit Rate@5).`

#### Gabarito de Decisão do Teste Pareado

| Modelo Local Candidato | Estatística $W$ (MRR@5) | p-valor (MRR@5) | Estatística $W$ (HR@5) | p-valor (HR@5) | Conclusão Estatística ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `qwen3_8b_q4km` | `346.0` | `0.9336` | `30.0` | `0.0029*` | `Equivalência estatística em MRR@5 (p > 0.05); Hit Rate@5 significativamente superior (p < 0.01)` |
| `qwen3_4b_q4km` | `468.5` | `0.9565` | `112.0` | `0.0343*` | `Equivalência estatística em MRR@5 (p > 0.05); Hit Rate@5 significativamente superior (p < 0.05)` |
| `bge_m3` | `355.5` | `0.8273` | `77.0` | `0.1266` | `Equivalência estatística em ambas as métricas (não rejeita H0)` |
| `multilingual_e5_large`| `276.0` | `0.5217` | `25.5` | `0.0124*` | `Equivalência estatística em MRR@5; Hit Rate@5 significativamente superior (p < 0.05)` |
| `nomic_embed_text` | `190.0` | `0.0142*` | `31.5` | `0.0017*` | `Significativamente inferior ao baseline em ambas as métricas (p < 0.05)` |
| `bertimbau_sts` | `562.5` | `0.0008*` | `351.0` | `0.7456` | `MRR@5 significativamente inferior ao baseline (p < 0.001)` |

*Critério: Se $p \ge 0.05$, a hipótese de diferença significante é rejeitada (não há evidência estatística de que o modelo local performa diferente do baseline da OpenAI).*

---

### 4.3 Intervalos de Confiança via Bootstrap (IC 95%)

`Os intervalos de confiança foram computados através de reamostragem não-paramétrica por Bootstrap com $B = 1.000$ iterações com reposição para o nível de confiança de 95%, garantindo robustez empírica frente à não-normalidade das métricas de IR.`

#### Intervalos Estimados com $B = 1.000$ Reamostragens

| Modelo | Hit Rate@5 [IC 95%] | MRR@5 [IC 95%] | Context Recall@5 [IC 95%] | MAP@5 [IC 95%] |
| :--- | :---: | :---: | :---: | :---: |
| `openai_3_small` (Base) | `[0.383, 0.555]` | `[0.305, 0.467]` | `[0.362, 0.534]` | `[0.531, 0.940]` |
| `qwen3_8b_q4km` | `[0.477, 0.648]` | `[0.309, 0.456]` | `[0.444, 0.618]` | `[0.506, 0.868]` |
| `qwen3_4b_q4km` | `[0.461, 0.633]` | `[0.307, 0.456]` | `[0.408, 0.582]` | `[0.482, 0.848]` |
| `bge_m3` | `[0.438, 0.609]` | `[0.304, 0.457]` | `[0.385, 0.556]` | `[0.547, 0.964]` |
| `multilingual_e5_large` | `[0.453, 0.633]` | `[0.325, 0.482]` | `[0.423, 0.603]` | `[0.482, 0.836]` |
| `nomic_embed_text` | `[0.273, 0.445]` | `[0.230, 0.376]` | `[0.263, 0.430]` | `[0.384, 0.738]` |
| `bertimbau_sts` | `[0.367, 0.539]` | `[0.187, 0.301]` | `[0.266, 0.433]` | `[0.234, 0.494]` |

---

### 4.4 Eficiência Computacional: Latência e Tempo de Indexação

`A eficiência operacional foi aferida medindo o tempo de construção integral do índice vetorial em disco e a latência média de recuperação top-10 por consulta sobre as 128 instâncias do conjunto de testes.`

#### Desempenho Operacional em Ambiente Local

| Modelo | Tempo de Indexação (s) | Latência Média de Consulta (ms) | Consultas por Segundo (QPS) | Tamanho do Índice em Disco (MB) |
| :--- | :---: | :---: | :---: | :---: |
| `openai_3_small` | `6.43 s` | `415.07 ms` | `2.41` | `4.4 MB` |
| `qwen3_8b_q4km` | `45.26 s` | `57.06 ms` | `17.52` | `13.0 MB` |
| `qwen3_4b_q4km` | `27.41 s` | `35.98 ms` | `27.79` | `7.7 MB` |
| `bge_m3` | `19.27 s` | `167.30 ms` | `5.98` | `3.2 MB` |
| `multilingual_e5_large` | `22.28 s` | `22.88 ms` | `43.71` | `3.4 MB` |
| `nomic_embed_text` | `11.95 s` | `9.78 ms` | `102.22` | `2.6 MB` |
| `bertimbau_sts` | `7.68 s` | `35.67 ms` | `28.03` | `2.7 MB` |

---

### 4.5 Visualizações Gráficas

Os gráficos correspondentes são gerados automaticamente pelo módulo `scripts/plot_embedding_graphs.py` e armazenados no diretório `graficos_tcc/`:

1. **Ranking por MRR@5 e Hit Rate@5**:
   `![Ranking por MRR@5 e Hit Rate@5](../graficos_tcc/ranking_hitrate_mrr_k5.png)`
   *(Localização: `graficos_tcc/ranking_hitrate_mrr_k5.png`)*
   
2. **Curva de Evolução do Recall por Top-K ($K \in \{2, 5, 10\}$)**:
   `![Curva de Recuperação por Top-K](../graficos_tcc/curva_recuperacao_topk.png)`
   *(Localização: `graficos_tcc/curva_recuperacao_topk.png`)*

3. **Trade-off de Engenharia: Eficiência (Latência em ms) vs. Eficácia (MRR@5)**:
   `![Trade-off Latência vs. Eficácia](../graficos_tcc/tradeoff_latencia_mrr.png)`
   *(Localização: `graficos_tcc/tradeoff_latencia_mrr.png`)*

---

## 5. Análise da Hipótese H1 e Discussão Científica

### 5.1 Matriz de Validação da Hipótese H1

A validação de $H_1$ exige que a razão relativa entre a métrica do modelo aberto e o baseline OpenAI seja $\ge 0.85$ (retenção $\ge 85\%$) em pelo menos duas métricas de IR no ponto de corte canônico $K=5$:

$$\text{Retenção}(\%) = \left( \frac{\text{Métrica}(M_{\text{candidato}})}{\text{Métrica}(M_{\text{OpenAI}})} \right) \times 100\%$$

| Modelo Candidato | % Retenção HR@5 | % Retenção MRR@5 | % Retenção Rec@5 | % Retenção MAP@5 | Critério Atendido (≥ 2 métricas ≥ 85%)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `qwen3_8b_q4km` | `121.65%` | `101.22%` | `118.50%` | `93.82%` | `SIM (4/4 métricas ≥ 85%)` |
| `qwen3_4b_q4km` | `118.32%` | `100.10%` | `110.78%` | `90.72%` | `SIM (4/4 métricas ≥ 85%)` |
| `bge_m3` | `111.65%` | `98.88%` | `104.26%` | `102.13%` | `SIM (4/4 métricas ≥ 85%)` |
| `multilingual_e5_large`| `116.66%` | `104.97%` | `114.01%` | `89.53%` | `SIM (4/4 métricas ≥ 85%)` |
| `nomic_embed_text` | `76.66%` | `78.67%` | `77.33%` | `76.10%` | `NÃO (0/4 métricas ≥ 85%)` |
| `bertimbau_sts` | `96.65%` | `63.10%` | `78.05%` | `49.68%` | `NÃO (Apenas 1 métrica ≥ 85%)` |

### 5.2 Discussão Detalhada dos Resultados

`A análise empírica dos dados consolidados traz evidências inequívocas quanto à sustentação da Hipótese Alternativa ($H_1$). Os dois modelos quantizados da família Qwen3 avaliados (`qwen3_8b_q4km` e `qwen3_4b_q4km`) demonstraram **retenção superior a 90% em todas as quatro métricas de IR avaliadas**, superando o baseline comercial proprietário da OpenAI em Hit Rate@5, MRR@5 e Recall@5.

Este comportamento comprova a teoria do **sweet spot de quantização** em k-quants (`Q4_K_M`). Diferente de quantizações uniformes ingênuas (RTN), o método k-quants preserva escalas críticas em blocos estatisticamente relevantes da matriz de projeção, retendo a curvatura e a topologia angular do espaço semântico em alta dimensionalidade (4096d para o Qwen3-8B e 2560d para o Qwen3-4B). A escala de parâmetros (7.6B e 4.0B) atua como um mecanismo compensador de altíssima eficácia: mesmo discretizado em 4 bits por peso, o modelo dispõe de uma capacidade representativa intrínseca ordens de magnitude superior a modelos de 100M a 500M de parâmetros em precisão completa FP16/FP32.

Adicionalmente, o modelo `multilingual-e5-large` ratificou sua excelência técnica em cenários de produção, conquistando o maior MRR@5 absoluto (0,4031), demonstrando que o pré-treinamento com prefixos contrastivos assimétricos (`query:` e `passage:`) organiza com precisão o ranqueamento imediato no primeiro posto ($k=1$). Em contraste marcante, o modelo monolíngue `bertimbau_sts` apresentou forte degradação de ordenação (MRR@5 de 0,2423, com queda de 36,9% em relação à OpenAI, $p = 0,0008$). Isso indica que a especialização puramente léxica em português sem alinhamento denso em pares massivos de perguntas e respostas é insuficiente para recuperação técnica especializada.`

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
  `Em ambientes de produção com interação em tempo real, a latência do pipeline RAG é um fator determinante para a usabilidade. Enquanto a API em nuvem da OpenAI incorre em um overhead de rede médio de 415,07 ms por consulta (sujeito a oscilações de conexão externa e fila de servidores), os modelos locais executados via barramento PCIe na GPU NVIDIA RTX PRO 1000 entregaram tempos de resposta substancialmente inferiores: 22,88 ms para o `multilingual-e5-large`, 35,98 ms para o `qwen3_4b_q4km` (ganho de 11,5×) e 57,06 ms para o `qwen3_8b_q4km` (ganho de 7,3×). Todos os candidatos locais operaram amplamente abaixo do teto de conforto de 150 ms estipulado para a etapa de recuperação vetorial.`
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

`A auditoria qualitativa manual foi executada sobre 20 amostras aleatórias sorteadas do conjunto de teste (`seed=42`). A avaliação identificou 17 amostras plenamente relevantes (85,0%), 1 parcialmente relevante (5,0%) e 2 amostras irrelevantes (10,0%) derivadas de ruídos de cabeçalho na extração PDF. Esse índice de 90,0% de aderência técnica direta atesta **Validade Alta e Baixo Viés Circular**, corroborando que os ganhos observados nos modelos abertos e quantizados refletem capacidade real de busca e não distorções sintéticas.`

#### Estrutura do Relatório de Auditoria das 20 Amostras

| ID Consulta | Pergunta Técnica | Resposta Esperada | Nó Recuperado (Modelo Aberto) | Avaliação Humana | Veredito |
| :---: | :--- | :--- | :--- | :--- | :---: |
| #01 | `Quais são as implicações do uso de comandos lineares na programação...` | `CM Comandos Lineares...` | `Cabeçalho isolado 'CM Comandos Lineares'` | `Ruído de extração (nome da empresa tomado como comando)` | `Irrelevante` |
| #02 | `Como o sistema Paralelo Multi Ativo muda a confiabilidade dos no-breaks?` | `SISTEMA PARALELO MULTI ATIVO (EXCLUSIVO CM COMANDOS)...` | `Trecho canônico sobre paralelismo redundante e sincronização de inversores` | `Resposta direta e tecnicamente precisa` | `Relevante` |
| ... | ... | ... | ... | ... | ... |
| #20 | `Quais são as dimensões físicas e a potência dos modelos?` | `Características Físicas e Mecânicas Dimensões Compactas Display TFT...` | `Tabela de especificações dimensionais e mecânicas da série corporativa` | `Casamento perfeito com parâmetros de engenharia` | `Relevante` |

### 7.2 Validade de Construto: Granularidade de Fragmentação (*Chunking*)

- O uso do `SentenceSplitter` com tamanho de 512 tokens e overlap de 50 tokens pode fragmentar tabelas de especificações técnicas elétricas (como curvas de rendimento, faixas de tensão de entrada/saída e capacidades de baterias).
- Como todos os 7 modelos foram submetidos exatamente à mesma divisão de nós em memória e em disco, a consistência comparativa foi mantida estritamente uniforme.

### 7.3 Validade Externa: Generalização do Domínio

- As conclusões empíricas deste benchmark são diretamente válidas para o domínio de **engenharia elétrica, automação e equipamentos de energia ininterrupta (nobreaks)** com documentação em português do Brasil.
- A generalização para corpora de domínio aberto ou textos puramente narrativos/literários não pode ser inferida sem experimentação complementar.

---

## 8. Conclusões e Recomendações

### 8.1 Conclusão Geral

`O presente estudo valida empiricamente que a transição de serviços comerciais proprietários em nuvem para modelos abertos locais de embedding não apenas é viável como é tecnicamente vantajosa no domínio de documentação técnica industrial. O modelo `qwen3-embedding:8b` quantizado em 4 bits (`Q4_K_M`) e o modelo `multilingual-e5-large` superaram a API da OpenAI em qualidade de recuperação, proporcionando ao mesmo tempo reduções drásticas na latência operacional (de 415 ms para 23-57 ms), custo zero de processamento e conformidade absoluta com requisitos corporativos de soberania de dados.`

### 8.2 Recomendações de Engenharia para o Sistema RAG

`Para a implementação definitiva no sistema de atendimento e suporte técnico da CM Comandos, recomenda-se a arquitetura baseada no modelo **`qwen3-embedding:4b` (Q4_K_M)** ou **`multilingual-e5-large`**.`

1. **Modelo Recomendado para Produção**: ``qwen3-embedding:4b` (Q4_K_M) ou `multilingual-e5-large``;
2. **Justificativa do Trade-off**: Balanço entre cobertura no Top-5 ($HR@5$), precisão de primeiro posto ($MRR@5$), latência em GPU e custo zero de infraestrutura;
3. **Estratégia de Deploy**: Implantação empacotada via container Docker utilizando o serviço Ollama para abstração de drivers de GPU e facilidade de substituição de modelos sem alteração no código da aplicação LlamaIndex.

### 8.3 Propostas de Trabalhos Futuros

`Recomenda-se expandir a bancada com busca híbrida densa-esparsa (BGE-M3 Sparse / BM25) combinada com re-ranking neural de dois estágios e avaliação ponta-a-ponta com geradores SLM locais (ex: Qwen 2.5 7B Instruct).`

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
