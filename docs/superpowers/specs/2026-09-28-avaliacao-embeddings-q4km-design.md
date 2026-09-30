# Especificação de Design: Avaliação Comparativa de Embeddings Quantizados (Q4_K_M) e Multilíngues vs. OpenAI

- **Data**: 2026-09-28
- **Branch**: `new_embeddings`
- **Ambiente de Destino para Execução**: OpenCode
- **Status**: ✅ Aprovado com Ressalvas (Revisão Opus 2026-09-28)

---

## 1. Visão Geral e Contexto

Este documento estabelece o desenho arquitetural e metodológico para o experimento comparativo de **modelos de embedding densos** aplicados à recuperação de informações em tarefas de *Retrieval-Augmented Generation* (RAG).

O objetivo central é avaliar se **modelos abertos quantizados em 4-bit (`Q4_K_M`)** e modelos abertos de destaque para a língua portuguesa (PT-BR) conseguem atingir paridade ou competitividade com o modelo de referência comercial proprietário **OpenAI `text-embedding-3-small`**, operando sobre um corpus técnico fechado da **CM Comandos Linha Nobre** (16 manuais e catálogos técnicos em PDF).

A avaliação é estritamente focada na **fase de recuperação (*Retriever-only*)**, eliminando a variabilidade, latência e custo computacional de geração de respostas por LLMs. A relevância dos contextos recuperados para diferentes valores de $K \in \{2, 5, 10\}$ é confrontada diretamente contra o gabarito sintetizado em `testset_openai_4omini.jsonl` (132 questões com `reference_contexts`).

---

## 2. Fundamentação Científica e o *Sweet Spot* do `Q4_K_M`

### 2.1 Justificativa Teórica e Prática do `Q4_K_M`
A quantização de representações densas é frequentemente alvo de questionamento quanto à perda de fidelidade semântica. No entanto, o método **k-quants** (introduzido por Iwan Kawrakow no ecossistema `llama.cpp` / GGUF) resolve as patologias de quantizações uniformes ingênuas (como `Q4_0` e `Q4_1`):

1. **Precisão Mista Adaptativa em Superblocos**: Na variante **`_M` (*Medium*)**, camadas de atenção sensíveis e tensores de projeção de feed-forward recebem alocação diferenciada (5 a 6 bits em sub-blocos críticos), enquanto tensores redundantes operam a 4 bits.
2. **Degradação Marginal do Espaço Latente (1% a 3%)**: Estudos pós-treinamento (Dettmers et al., 2023 [*QLoRA*]; Frantar et al., 2022 [*GPTQ*]) demonstram que esquemas de 4 bits baseados em blocos preservam a geometria angular das representações vetoriais. A perplexidade ($\Delta \text{ppl}$) aumenta tipicamente em menos de 0.1, e a similaridade de cosseno angular sofre perturbação uniforme desprezível para ordenação de ranking.
3. **Ponto Ótimo de Equilíbrio (*Sweet Spot*)**:
   - Redução de ~75% de pegada de memória/VRAM e volume de armazenamento em comparação ao FP16.
   - O modelo `qwen3-embedding:8b` (7.6B parâmetros) reduz-se a apenas 4.7 GB de VRAM, viabilizando execução local rápida com latência baixa.
   - Variantes com maior densidade de bits (como `Q8_0`) dobram a exigência de memória com retornos decrescentes (ganhos observados inferiores a 0.5% em tarefas de IR), justificando o foco metodológico em `Q4_K_M`.

### 2.2 Hipóteses do Experimento
- **Hipótese de Trabalho ($H_1$)**: Pelo menos um dos modelos abertos locais (`Qwen3-8B-Q4_K_M` ou `BGE-M3`) atinge $HitRate@5 \ge 0.85 \times HitRate@5_{\text{OpenAI}}$ e $MRR@5 \ge 0.85 \times MRR@5_{\text{OpenAI}}$, comprovando viabilidade técnica para substituição de APIs em RAG técnico em português.
- **Hipótese Nula ($H_0$)**: A quantização em `Q4_K_M` ou as restrições linguísticas dos modelos abertos acarretam uma degradação estatisticamente significante ($p < 0.05$) superior a 15% nas métricas de recuperação em relação à referência OpenAI.

---

## 3. Matriz de Modelos de Embedding Avaliados

A suíte experimental é composta por **7 modelos representativos**:

| ID | Nome do Modelo | Provedor / Mecanismo | Dimensão Vetorial | Contexto Máx. | Justificativa no Experimento |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **M1** | `text-embedding-3-small` | OpenAI API (Chave via `.env`) | 1536 | 8191 | **Baseline de Referência**: Padrão proprietário de mercado amplamente utilizado no setor. |
| **M2** | `qwen3-embedding:8b` | Ollama Local (`Q4_K_M`) | 4096 | 40960 | **Candidato Principal Q4**: 7.6B parâmetros quantizado, testando a capacidade de modelos densos de grande porte em 4-bit. |
| **M3** | `qwen3-embedding:4b` | Ollama Local (`Q4_K_M`) | 2560 | 40960 | **Escalabilidade Q4**: 4.0B parâmetros quantizado, avaliando o impacto da redução de escala em 4-bit. |
| **M4** | `bge-m3` | Ollama / HuggingFace (`BAAI/bge-m3`) | 1024 | 8192 | **Padrão Ouro Multilíngue Aberto**: Reconhecido como topo de linha no MTEB para recuperação multilíngue, com suporte nativo robusto a PT-BR. |
| **M5** | `multilingual-e5-large` | HuggingFace (`intfloat/multilingual-e5-large`) | 1024 | 512 | **Referência de Produção Acadêmica**: Utilizado previamente no pipeline `haiku-rag` com histórico documentado. |
| **M6** | `nomic-embed-text` | Ollama Local | 768 | 8192 | **Eficiência Extrema**: Modelo ultraleve (~137M parâmetros) com janela de 8k tokens para avaliar trade-off de velocidade. |
| **M7** | `bertimbau-base-portuguese-sts` | HuggingFace (`ricardoz/bertimbau-base-portuguese-sts`) | 768 | 512 | **Especialista em PT-BR**: Treinado sobre o corpus brasileiro BrWaC e ajustado para Semantic Textual Similarity (STS). |

---

## 4. Corpus de Dados e Ground Truth

### 4.1 Corpus de Documentos (`data/`)
- 16 catálogos e manuais técnicos da **CM Comandos Linha Nobre** em formato PDF (ex: `banco-de-baterias-cuidados.pdf`, `catalogo-dominion-sp1-1.pdf`, `catalogo-sistema-equalizer.pdf`, etc.).
- Conteúdo: especificações elétricas, diagramas operacionais, termos técnicos em português sobre no-breaks, retificadores, inversores e sistemas de energia crítica.

### 4.2 Ground Truth (`testset_openai_4omini.jsonl`)
- 132 amostras sintéticas validadas criadas pelo framework Ragas (`TestsetGenerator`) via GPT-4o-mini.
- Cada entrada $q$ contém:
  - `user_input`: pergunta do usuário em linguagem natural (português).
  - `reference_contexts`: lista de trechos literais dos documentos que sustentam a resposta correta ($\mathcal{R}_q$).
  - `reference`: resposta ground-truth esperada.
  - `synthesizer_name`: tipo de sintetizador (`AbstractQuerySynthesizer`, `SpecificQuerySynthesizer`, etc.).

---

## 5. Arquitetura de Isolamento dos Índices e Execução

### 5.1 Isolamento de Persistência (`storage/`)
Para garantir que cada modelo mantenha seu espaço vetorial isolado e permitir que o experimento seja reexecutado sem custo adicional de API:
```text
storage/
├── index_openai_3_small/
├── index_qwen3_8b_q4km/
├── index_qwen3_4b_q4km/
├── index_bge_m3/
├── index_multilingual_e5_large/
├── index_nomic_embed_text/
└── index_bertimbau_sts/
```

### 5.2 Controle de Chunking
Para assegurar a validade interna do experimento, o particionamento dos documentos deve ser rigorosamente idêntico para todos os 7 modelos:
- Algoritmo: `SentenceSplitter` do LlamaIndex.
- `chunk_size`: 512 tokens.
- `chunk_overlap`: 50 tokens.
- Separadores: quebras de parágrafo e pontuação padrão.

### 5.3 Download dos Modelos e Aceleração via Hardware (NVIDIA CUDA / GPU)

O ambiente de execução dispõe de GPU **NVIDIA RTX PRO 1000 (Blackwell)** com **~8 GB de VRAM** (CUDA 13.4). Todos os modelos locais devem ser executados com aceleração total em GPU para maximizar a taxa de transferência e minimizar a latência de indexação e recuperação.

#### 1. Download e Execução dos Modelos Ollama (GPU Offload Total)
Os seguintes modelos devem ser baixados e verificados no Ollama:
```bash
# 1. Candidato Principal Q4_K_M (7.6B, ~4.7 GB - cabe 100% na VRAM)
ollama pull qwen3-embedding:8b

# 2. Candidato Leve Q4_K_M (4.0B, ~2.5 GB)
ollama pull qwen3-embedding:4b

# 3. Referência Multilíngue Aberta (~2.2 GB)
ollama pull bge-m3

# 4. Ultraleve Eficiente (~274 MB)
ollama pull nomic-embed-text
```
*Configuração de GPU no Ollama*:
- O Ollama carrega automaticamente 100% das camadas na GPU (`ngl` / `gpu_layers = 100%`).
- No LlamaIndex, instanciar via `OllamaEmbedding(model_name="...", base_url="http://localhost:11434")`.

#### 2. Pré-carregamento e Execução dos Modelos HuggingFace na GPU (`device="cuda"`)
Para os modelos HuggingFace (`multilingual-e5-large` e `bertimbau-base-portuguese-sts`), o download é realizado no cache local (`~/.cache/huggingface/`) e a inferência deve ser explicitamente roteada para CUDA:
```python
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

device = "cuda" if torch.cuda.is_available() else "cpu"

# Multilingual E5 Large na GPU
embed_e5 = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large",
    device=device,
    embed_batch_size=32
)

# Bertimbau STS na GPU
embed_bertimbau = HuggingFaceEmbedding(
    model_name="ricardoz/bertimbau-base-portuguese-sts",
    device=device,
    embed_batch_size=32
)
```
*Pré-download via CLI (opcional para execução offline no OpenCode)*:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large'); SentenceTransformer('ricardoz/bertimbau-base-portuguese-sts')"
```

---

## 6. Mecanismo de Avaliação e Matching de Recuperação

### 6.1 Execução da Busca
Para cada modelo $m \in \{M_1, \dots, M_7\}$:
1. Instanciar o retriever com $K_{\max} = 10$:
   ```python
   retriever = index.as_retriever(similarity_top_k=10)
   ```
2. Para cada questão $q \in \text{testset}$:
   - Coletar lista ordenada de nós: $[N_1, N_2, \dots, N_{10}]$.
   - Extrair metadados, texto do nó e similaridade de cosseno.

### 6.2 Critérios Objetivos de Relevância (Matching com $\mathcal{R}_q$)
Um nó recuperado $N_j$ é classificado como **Relevante ($\text{IsRelevant} = 1$)** para a consulta $q$ se satisfizer **pelo menos um** dos critérios abaixo em relação a qualquer contexto de referência $r \in \mathcal{R}_q$:
1. **Contenção Estrita**: $r \subset N_j.\text{text}$ ou $N_j.\text{text} \subset r$ (para trechos superiores a 40 caracteres).
2. **Fuzzy Token Matching**: $\text{RapidFuzz.token\_set\_ratio}(N_j.\text{text}, r) \ge 75.0$.
3. **ROUGE-L Overlap**: $F_1 \ge 0.50$ calculado pela biblioteca `rouge-score`.

### 6.3 Formulação Matemática das Métricas
Para cada modelo $m$ e valor de $K \in \{2, 5, 10\}$:

1. **Hit Rate@K**:
   $$\text{HitRate@K} = \frac{1}{|Q_{\text{efetivo}}|} \sum_{q \in Q_{\text{efetivo}}} \mathbb{I}\left(\exists j \le K : N_j \text{ é relevante}\right)$$

2. **Context Recall@K**:
   $$\text{Recall@K} = \frac{1}{|Q_{\text{efetivo}}|} \sum_{q \in Q_{\text{efetivo}}} \frac{\left|\{r \in \mathcal{R}_q \mid \exists j \le K, N_j \text{ casa com } r\}\right|}{|\mathcal{R}_q|}$$

3. **Mean Reciprocal Rank (MRR@K)**:
   $$\text{MRR@K} = \frac{1}{|Q_{\text{efetivo}}|} \sum_{q \in Q_{\text{efetivo}}} \left( \frac{1}{\min \{j \le K \mid N_j \text{ é relevante}\}} \quad \text{se houver acerto, senão } 0 \right)$$

4. **Mean Average Precision (MAP@K)**:
   $$\text{MAP@K} = \frac{1}{|Q_{\text{efetivo}}|} \sum_{q \in Q_{\text{efetivo}}} \frac{1}{\min(|\mathcal{R}_q|, K)} \sum_{j=1}^{K} \text{Precision}@j \cdot \text{Rel}(N_j)$$

5. **Significância Estatística (Teste de Wilcoxon Signed-Rank & Bootstrap CI 95%)**:
   - Cada modelo aberto é comparado par a par contra o baseline OpenAI para $K=5$.
   - O script `consolidate_metrics.py` reporta a média, o intervalo de confiança de 95% via bootstrap ($B=1000$) e o valor-p do teste de Wilcoxon assinalado, determinando se diferenças observadas são estatisticamente significativas ($p < 0.05$).

---

## 7. Diretrizes de Engenharia e Operação no OpenCode

### 7.1 Gestão de VRAM e Execução Sequencial Isolada (RTX PRO 1000)
- Como a GPU possui 8 GB de VRAM e o modelo `qwen3-embedding:8b` (Q4_K_M) ocupa ~4.7 GB:
  - Os modelos devem ser avaliados de forma **estritamente sequencial**.
  - No Ollama, definir `OLLAMA_KEEP_ALIVE=0` (ou chamada explícita `ollama stop <model>`) antes de carregar o modelo subsequente para evitar retenção de memória e erro de Out-Of-Memory (OOM).
  - Limpeza de cache CUDA entre modelos via `torch.cuda.empty_cache()` e `gc.collect()`.

### 7.2 Tratamento Obrigatório de Prefixos para E5 (`multilingual-e5-large`)
- O modelo E5 exige assimetria de prefixos:
  - Documentos / Chunks indexados: prefixados com `passage: `.
  - Perguntas / Consultas de busca: prefixadas com `query: `.
  - Implementado explicitamente no adapter `HuggingFaceEmbedding` com `text_instruction="passage: "` e `query_instruction="query: "`.

### 7.3 Saneamento do Testset e Reprodutibilidade
- **Deduplicação**: O script carrega `testset_openai_4omini.jsonl`, remove registros duplicados com base em `(user_input, reference_contexts)`, registrando e reportando o tamanho efetivo $N_{\text{efetivo}}$ no relatório.
- **Sementes Fixas**: Configuração determinística com `seed = 42` (`random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`).

### 7.4 Mitigação Epistemológica: Viés Circular do Ground Truth
- O ground truth sintetizado originalmente com GPT-4o-mini e embeddings OpenAI possui viés intrínseco que tende a favorecer a referência proprietária.
- **Mitigação Metodológica**:
  1. O relatório técnico formaliza esta limitação na seção de Ameaças à Validade Interna.
  2. Implementação de uma rotina de auditoria amostral qualitativa (~15% do testset) para registrar casos em que modelos locais recuperam trechos tecnicamente válidos do manual que não constavam no recorte estrito do gabarito sintetizado.

### 7.5 Módulos a Serem Implementados
1. **`scripts/benchmark_embeddings.py`**:
   - CLI configurável (`--models all`, `--top_k 2,5,10`, `--rebuild_index false`).
   - Carrega chaves de `.env`.
   - Inicializa LlamaIndex com os adaptadores de embedding apropriados (`OpenAIEmbedding`, `OllamaEmbedding`, `HuggingFaceEmbedding`).
   - Executa buscas e salva resultados brutos em `results/raw_retrievals_<model>.json`.
2. **`scripts/consolidate_metrics.py`**:
   - Lê os arquivos brutos, aplica as regras de matching e gera `results/consolidated_embeddings_metrics.csv`.
   - Gera tabela Markdown comparativa para inclusão direta no relatório de pesquisa (ICT)/artigo.
3. **`scripts/plot_embedding_graphs.py`**:
   - Gera gráficos científicos com Seaborn/Matplotlib salvos em `graficos_tcc/`:
     - Ranking de Hit Rate@5 e MRR@5 por modelo.
     - Curva de recuperação por $K$ (Recall@2 vs Recall@5 vs Recall@10).
     - Gráfico de dispersão Eficiência (latência) vs Eficácia (MRR@5).

### 7.6 Validação do Ambiente e Dependências
- `requirements.txt` atualizado garantindo:
  - `llama-index-core`
  - `llama-index-embeddings-openai`
  - `llama-index-embeddings-ollama`
  - `llama-index-embeddings-huggingface`
  - `ragas==0.2.3`
  - `RapidFuzz>=3.11.0`
  - `rouge-score>=0.1.2`
  - `seaborn` / `matplotlib` / `pandas`
  - `torch>=2.4.0` / `sentence-transformers>=3.0.0`

---

## 8. Protocolo de Validação com Agente Opus

Antes do handoff final para o OpenCode:
1. Uma rodada de revisão técnica e de consistência do plano será executada invocando o Claude Opus via CLI local (`claude -p "..." --model opus`).
2. O agente Opus verificará:
   - Coerência das fórmulas matemáticas.
   - Robustez contra erros de limites de contexto e timeouts do Ollama.
   - Completude dos scripts para execução não-interativa no OpenCode.
