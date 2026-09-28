# Plano de Implementação: Avaliação Comparativa de Embeddings Quantizados (Q4_K_M) e Multilíngues vs. OpenAI

> **Para executores no OpenCode / agentes:** SUB-SKILL REQUERIDA: Execute este plano de forma incremental tarefa por tarefa. As etapas utilizam a sintaxe de checkbox (`- [ ]`) para rastreamento de progresso.

**Objetivo:** Implementar e executar o pipeline completo de benchmark de recuperação (*Retriever-only*), comparando modelos locais quantizados em 4-bit (`Q4_K_M`) e multilíngues abertos contra a referência proprietária OpenAI `text-embedding-3-small` sobre o acervo técnico da CM Comandos, gerando métricas formais de IR, testes de significância estatística e gráficos para o TCC/artigo.

**Arquitetura:** Pipeline modular em Python executado headless via CLI. Utiliza LlamaIndex com isolamento de índices em disco por modelo (`storage/index_<id>/`), matching multicamada determinístico contra ground truth (substring, RapidFuzz e ROUGE-L), gestão sequencial de VRAM na GPU NVIDIA RTX PRO 1000 com descarga explícita entre modelos, cálculo de significância estatística (teste pareado de Wilcoxon e Bootstrap CI 95%) e renderização de figuras científicas em `graficos_tcc/`.

**Tech Stack:** Python 3.12, LlamaIndex, Ollama (CUDA), PyTorch, HuggingFace SentenceTransformers, RapidFuzz, rouge-score, SciPy, Pandas, Seaborn/Matplotlib.

**Spec:** [docs/superpowers/specs/2026-09-28-avaliacao-embeddings-q4km-design.md](file:///home/joaorura/orca/workspaces/rag_eval/gorgonian/docs/superpowers/specs/2026-09-28-avaliacao-embeddings-q4km-design.md)  
**Metodologia Científica:** [docs/metodologia_experimento_embeddings.md](file:///home/joaorura/orca/workspaces/rag_eval/gorgonian/docs/metodologia_experimento_embeddings.md)  
**Especificação dos Modelos:** [docs/especificacao_modelos_embeddings.md](file:///home/joaorura/orca/workspaces/rag_eval/gorgonian/docs/especificacao_modelos_embeddings.md)

---

## Restrições Globais

- Todos os modelos operam sob chunking idêntico: `SentenceSplitter(chunk_size=512, chunk_overlap=50)`.
- Chave da OpenAI extraída via `.env` espelhada de `/home/joaorura/Desenvolvimento/EASY/cm-comandos-piad/.env`.
- Execução sequencial mandatória na GPU para evitar saturação da VRAM (limite de 8 GB).
- Sementes aleatórias fixadas em `42` (`random`, `numpy`, `torch`).
- Modelo `multilingual-e5-large` obrigatoriamente configurado com prefixo `passage: ` para chunks e `query: ` para buscas.
- Testset saneado com remoção de duplicatas antes do cálculo das métricas.

---

## Foco de Revisão e Casos de Falha Prevenidos

1. **OOM na GPU**: Carregamento de múltiplos modelos na VRAM simultaneamente. *Prevenção*: `OLLAMA_KEEP_ALIVE=0`, `torch.cuda.empty_cache()` e `gc.collect()` antes de alternar de modelo.
2. **Degradação Silenciosa do E5**: Omissão dos prefixos assimétricos `query:` / `passage:`. *Prevenção*: Teste unitário verificando os prefixos injetados.
3. **Falso Negativo por Borda de Chunks**: Desalinhamento de quebra de página/parágrafo. *Prevenção*: Critério OR (substring OU RapidFuzz $\ge 75\%$ OU ROUGE-L $F_1 \ge 0.50$).
4. **Viés Circular de Ground Truth**: Overfitting metodológico ao baseline M1. *Prevenção*: Auditoria amostral qualitativa e documentação formal em ameaças à validade.
5. **Divergência de Denominador por Duplicatas**: Linhas idênticas inflando o N. *Prevenção*: Deduplicação preliminar registrando o tamanho efetivo $N_{\text{efetivo}}$.

---

### Tarefa 1: Configuração do Ambiente e Sincronização do `.env`

**Arquivos:**
- Modificar: `requirements.txt`
- Criar: `.env` (espelhando a chave `OPENAI_API_KEY`)
- Testar: `scripts/verify_env.py`

- [ ] **Passo 1.1: Criar o arquivo `.env` sincronizado**
  Extrair a variável `OPENAI_API_KEY` de `/home/joaorura/Desenvolvimento/EASY/cm-comandos-piad/.env` e gravá-la em `./.env`.
  Comando de verificação: `grep OPENAI_API_KEY .env | grep -v "=$"`

- [ ] **Passo 1.2: Instalar/Atualizar as dependências no ambiente Python**
  Executar: `pip install -r requirements.txt`

- [ ] **Passo 1.3: Criar script de verificação de sanidade do ambiente**
  Criar `scripts/verify_env.py` checando:
  - Presença e validade de `OPENAI_API_KEY`.
  - Disponibilidade de CUDA via `torch.cuda.is_available()` e nome da GPU (`NVIDIA RTX PRO 1000`).
  - Conexão HTTP com o Ollama em `http://localhost:11434`.
  - Executar: `python scripts/verify_env.py`

- [ ] **Passo 1.4: Commit**
  `git add .env requirements.txt scripts/verify_env.py && git commit -m "chore: setup de ambiente e script de verificação"`

---

### Tarefa 2: Download e Disponibilização dos Modelos Locais (CUDA)

**Arquivos:**
- Criar: `scripts/download_models.sh`

- [ ] **Passo 2.1: Criar script de download e pré-cache de modelos**
  Criar `scripts/download_models.sh` com:
  ```bash
  #!/usr/bin/env bash
  set -e
  echo "=== 1. Verificando e baixando modelos Ollama ==="
  ollama pull qwen3-embedding:8b
  ollama pull qwen3-embedding:4b
  ollama pull bge-m3
  ollama pull nomic-embed-text

  echo "=== 2. Pré-carregando modelos HuggingFace no cache local ==="
  python -c "
  from sentence_transformers import SentenceTransformer
  print('Baixando intfloat/multilingual-e5-large...')
  SentenceTransformer('intfloat/multilingual-e5-large')
  print('Baixando ricardoz/bertimbau-base-portuguese-sts...')
  SentenceTransformer('ricardoz/bertimbau-base-portuguese-sts')
  print('Modelos HuggingFace prontos!')
  "
  ```

- [ ] **Passo 2.2: Executar o download dos modelos**
  Executar: `bash scripts/download_models.sh`
  Verificar no Ollama: `ollama list | grep -E "qwen3|bge-m3|nomic"`

- [ ] **Passo 2.3: Commit**
  `git add scripts/download_models.sh && git commit -m "feat: script de download e pre-caching de modelos"`

---

### Tarefa 3: Mecanismo de Matching e Saneamento do Testset

**Arquivos:**
- Criar: `scripts/matching_engine.py`
- Criar: `tests/test_matching_engine.py`

- [ ] **Passo 3.1: Escrever teste unitário para o matching engine**
  Criar `tests/test_matching_engine.py` testando:
  - Deduplicação de dataset (linhas idênticas removidas).
  - Casamento estrito por substring.
  - Casamento fuzzy com `RapidFuzz` (token_set_ratio $\ge 75\%$).
  - Casamento via ROUGE-L ($F_1 \ge 0.50$).
  - Caso negativo (texto completamente disconexo retorna `False`).
  Executar: `pytest tests/test_matching_engine.py` (deve falhar pois o módulo ainda não existe).

- [ ] **Passo 3.2: Implementar o `scripts/matching_engine.py`**
  Implementar:
  - Função `load_and_deduplicate_testset(jsonl_path: str) -> tuple[pd.DataFrame, int, int]`.
  - Função `is_context_relevant(node_text: str, reference_contexts: list[str]) -> tuple[bool, str]`.
  - Função `compute_query_retrieval_metrics(retrieved_nodes: list[dict], reference_contexts: list[str], k_values: list[int]) -> dict`.

- [ ] **Passo 3.3: Executar testes unitários do matching**
  Executar: `pytest tests/test_matching_engine.py`
  Garantir 100% de aprovação.

- [ ] **Passo 3.4: Commit**
  `git add scripts/matching_engine.py tests/test_matching_engine.py && git commit -m "feat: matching engine de recuperação e deduplicação de testset"`

---

### Tarefa 4: Script Principal de Benchmark de Recuperação (`benchmark_embeddings.py`)

**Arquivos:**
- Criar: `scripts/benchmark_embeddings.py`

- [ ] **Passo 4.1: Implementar o Factory de Embeddings com suporte a GPU e prefixos**
  Função `get_embedding_model(model_id: str)` instanciando:
  - `openai_3_small`: `OpenAIEmbedding(model="text-embedding-3-small", api_key=...)`.
  - `qwen3_8b_q4km`: `OllamaEmbedding(model_name="qwen3-embedding:8b", base_url="http://localhost:11434")`.
  - `qwen3_4b_q4km`: `OllamaEmbedding(model_name="qwen3-embedding:4b", base_url="http://localhost:11434")`.
  - `bge_m3`: `OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")`.
  - `nomic_embed_text`: `OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")`.
  - `multilingual_e5_large`: `HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large", device="cuda", text_instruction="passage: ", query_instruction="query: ")`.
  - `bertimbau_sts`: `HuggingFaceEmbedding(model_name="ricardoz/bertimbau-base-portuguese-sts", device="cuda")`.

- [ ] **Passo 4.2: Implementar rotina de indexação isolada e persistente**
  - Criação da pasta `storage/index_<model_id>/`.
  - Se existir: carrega com `StorageContext.from_defaults(persist_dir=...)` e `load_index_from_storage`.
  - Se não existir: carrega `data/` com `SimpleDirectoryReader`, divide com `SentenceSplitter(chunk_size=512, chunk_overlap=50)`, constrói `VectorStoreIndex.from_documents()` e persiste.
  - Registra tempo de indexação e tamanho em disco.

- [ ] **Passo 4.3: Implementar o loop de recuperação e gestão de VRAM**
  - Para cada modelo:
    - Recuperar Top-10 para as consultas do testset saneado.
    - Medir latência por query.
    - Salvar resultados brutos em `results/raw_retrievals_<model_id>.json`.
    - **Gestão de VRAM**: Descarregar modelo Ollama (`curl http://localhost:11434/api/generate -d '{"model": "<nome>", "keep_alive": 0}'`), executar `torch.cuda.empty_cache()` e `gc.collect()`.

- [ ] **Passo 4.4: Adicionar interface de linha de comando (CLI)**
  Suportar argumentos:
  - `--models` (ex: `all` ou lista separada por vírgula).
  - `--limit` (para rodar smoke tests com 5 queries).
  - `--rebuild_index` (flag booleana para forçar regeração).

- [ ] **Passo 4.5: Executar Smoke Test (5 consultas)**
  Executar: `python scripts/benchmark_embeddings.py --models qwen3_4b_q4km --limit 5`
  Verificar se o JSON de resultados em `results/raw_retrievals_qwen3_4b_q4km.json` foi gerado perfeitamente.

- [ ] **Passo 4.6: Commit**
  `git add scripts/benchmark_embeddings.py && git commit -m "feat: script central de benchmark de embeddings com gestao de VRAM"`

---

### Tarefa 5: Consolidação de Métricas e Testes de Significância Estatística

**Arquivos:**
- Criar: `scripts/consolidate_metrics.py`
- Criar: `tests/test_statistics.py`

- [ ] **Passo 5.1: Escrever testes para o cálculo estatístico**
  Testar cálculo de HR@K, MRR@K, MAP@K e o teste de Wilcoxon contra um ground truth sintético com diferenças conhecidas.
  Executar: `pytest tests/test_statistics.py`

- [ ] **Passo 5.2: Implementar `scripts/consolidate_metrics.py`**
  Implementar:
  - Leitura de todos os arquivos `results/raw_retrievals_*.json`.
  - Cálculo de $HitRate@K$, $Recall@K$, $MRR@K$ e $MAP@K$ para $K \in \{2, 5, 10\}$.
  - Cálculo de médias, desvios-padrão e Intervalos de Confiança (Bootstrap 95%, $B=1000$).
  - Teste de Wilcoxon pareado para cada modelo vs. `openai_3_small` no $K=5$, reportando o valor-$p$.
  - Exportação de `results/consolidated_embeddings_metrics.csv`.
  - Geração de tabela formatada em Markdown no terminal e no arquivo `results/summary_table.md`.

- [ ] **Passo 5.3: Validar consolidação**
  Executar: `python scripts/consolidate_metrics.py`
  Verificar se o CSV e a tabela Markdown são gerados corretamente.

- [ ] **Passo 5.4: Commit**
  `git add scripts/consolidate_metrics.py tests/test_statistics.py && git commit -m "feat: consolidacao de metricas e teste pareado de Wilcoxon"`

---

### Tarefa 6: Visualização Gráfica Científica para TCC/Artigo

**Arquivos:**
- Criar: `scripts/plot_embedding_graphs.py`

- [ ] **Passo 6.1: Implementar o gerador de figuras com Seaborn/Matplotlib**
  Gera 3 gráficos salvos em `graficos_tcc/`:
  1. `ranking_hitrate_mrr_k5.png`: Gráfico de barras horizontais ordenado por MRR@5 com barras de erro (IC 95%), destacando o baseline OpenAI e os modelos Q4_K_M.
  2. `curva_recuperacao_topk.png`: Gráfico de linhas mostrando a evolução de Recall@2 -> Recall@5 -> Recall@10 para os 7 modelos.
  3. `tradeoff_latencia_mrr.png`: Gráfico de dispersão Eficiência (latência em ms/query no eixo X) vs. Eficácia (MRR@5 no eixo Y), com quadrantes de viabilidade.

- [ ] **Passo 6.2: Testar a geração de gráficos**
  Executar: `python scripts/plot_embedding_graphs.py`
  Verificar se as imagens PNG são salvas com resolução de 300 DPI em `graficos_tcc/`.

- [ ] **Passo 6.3: Commit**
  `git add scripts/plot_embedding_graphs.py && git commit -m "feat: gerador de graficos de publicacao para o TCC"`

---

### Tarefa 7: Execução Completa e Relatório Técnico Final no OpenCode

**Arquivos:**
- Criar: `docs/analise_comparativa_resultados.md`

- [ ] **Passo 7.1: Executar o benchmark completo para os 7 modelos**
  Executar: `python scripts/benchmark_embeddings.py --models all`
  Monitorar via terminal a indexação e a execução sequencial sem estourar os 8 GB de VRAM.

- [ ] **Passo 7.2: Consolidar as métricas finais e gráficos**
  Executar:
  ```bash
  python scripts/consolidate_metrics.py
  python scripts/plot_embedding_graphs.py
  ```

- [ ] **Passo 7.3: Redigir o relatório final de análise comparativa**
  Criar `docs/analise_comparativa_resultados.md` contendo:
  - Tabela consolidada com todas as métricas para $K \in \{2, 5, 10\}$.
  - Análise da hipótese $H_1$ (se Q4_K_M atingiu $\ge 85\%$ do desempenho da OpenAI).
  - Resultados do teste de Wilcoxon ($p$-values).
  - Análise de viabilidade prática (tamanho de VRAM, velocidade e custo zero de API).
  - Relatório da auditoria amostral qualitativa mitiga o viés de circularidade.

- [ ] **Passo 7.4: Commit final da branch**
  `git add results/ docs/analise_comparativa_resultados.md graficos_tcc/ && git commit -m "feat: resultados consolidados do benchmark de embeddings"`
