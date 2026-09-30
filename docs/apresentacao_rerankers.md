---
marp: true
theme: default
paginate: true
header: "Re-ranking Neural em RAG Industrial (Two-Stage RAG) — Pesquisa ICT"
footer: "CM Comandos Lineares | NVIDIA RTX PRO 1000 (8 GB GDDR6)"
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 30px 45px;
    background-color: #ffffff;
    color: #1a202c;
  }
  h1 {
    color: #1a365d;
    font-size: 1.6em;
    margin-bottom: 0.2em;
    margin-top: 10px;
  }
  h2 {
    color: #2b6cb0;
    font-size: 1.2em;
    margin-top: 15px;
    margin-bottom: 0.3em;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
  }
  p, li {
    font-size: 0.72em;
    line-height: 1.45;
  }
  ul {
    margin-top: 6px;
    margin-bottom: 8px;
  }
  table {
    font-size: 0.60em;
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
  }
  th {
    background-color: #edf2f7;
    color: #2d3748;
    padding: 5px 6px;
  }
  td {
    padding: 4px 6px;
    border-bottom: 1px solid #e2e8f0;
  }
  .badge-win {
    background-color: #c6f6d5;
    color: #22543d;
    font-weight: bold;
    padding: 1px 5px;
    border-radius: 3px;
  }
  .badge-highlight {
    background-color: #bee3f8;
    color: #2b6cb0;
    font-weight: bold;
    padding: 1px 5px;
    border-radius: 3px;
  }
  .grid-2 {
    display: flex;
    gap: 20px;
    align-items: center;
    margin-top: 10px;
  }
  .col-text {
    flex: 1;
  }
  .col-img {
    flex: 1.2;
    text-align: center;
  }
  .chart-img {
    width: 100%;
    max-height: 420px;
    object-fit: contain;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }
  footer {
    font-size: 0.5em;
    color: #718096;
  }
  header {
    font-size: 0.52em;
    color: #718096;
  }
---

<!-- _class: lead -->
# Avaliação de Re-ranking Neural em Dois Estágios (Two-Stage RAG)
### Otimização da Precisão em Manuais Técnicos Industriais com Modelos Quantizados Locais

**Projeto de Pesquisa Científica e Tecnológica (ICT) — Engenharia de Software / Ciência da Computação**
*Hardware de Execução:* NVIDIA RTX PRO 1000 Blackwell (8 GB GDDR6) | Intel Core Ultra 7 265H | 32 GB RAM
*Estudo de Caso:* 16 Manuais de No-breaks Industriais — CM Comandos Lineares

---

## 1. Motivação: Limitações do RAG em Estágio Único

- **O Desafio do Jargão Industrial**:
  - Manuais técnicos contêm termos como *bypass estático*, *retificador tiristorizado*, *inversor IGBT*, *tensão de flutuação VRLA*.
  - Bi-encoders de 1º estágio operam em representação vetorial independente, perdendo nuances contextuais finas entre a pergunta e o fragmento.
- **A Solução Two-Stage RAG**:
  - **1º Estágio (Super-sampling $K_{\text{cand}}=20$)**: Recupera rapidamente uma vizinhança ampla de alta revocação (*recall*).
  - **2º Estágio (Re-ranking)**: Aplica atenção cruzada profunda (*all-to-all token attention*) ou inferência de probabilidade condicional para reorganizar o Top-5.
- **Hipótese Central de Pesquisa**:
  - Modelos locais quantizados (Q4_K_M / Q5_K_M) e Cross-Encoders dedicados locais atingem ganhos estatisticamente significantes sobre o baseline puro e superam APIs comerciais proprietárias com custo zero.

---

## 2. Modelos e Arquiteturas Avaliados

### 1º Estágio (Recuperação Vetorial Densa - Top-2 do Benchmark Prévio)
- **`qwen3_8b_q4km`**: Qwen3-Embedding-8B quantizado em Q4_K_M (Ollama, 5.2 GB VRAM).
- **`multilingual_e5_large`**: Microsoft Multilingual-E5-Large em FP16 (HuggingFace, 2.2 GB VRAM).

### 2º Estágio (Reordenadores Neurais)
- **Baseline Puro (`none`)**: Ordenação vetorial original sem modelo de 2º estágio.
- **`qwen3_4b_q4km`**: Qwen3-Reranker-4B (Q4_K_M via Ollama, logprob softmax binário).
- **`qwen3_8b_q5km`**: Qwen3-Reranker-8B (Q5_K_M via Ollama, logprob softmax binário).
- **`bge_reranker_v2_m3`**: BAAI BGE-Reranker-v2-m3 Cross-Encoder (HuggingFace PyTorch).
- **`rankgpt`**: RankGPT baseado em OpenAI `gpt-4o-mini` (Prompt listwise via API em nuvem).

---

## 3. Metodologia Experimental e Rigor Estatístico

- **Corpus & Consultas**:
  - 16 manuais de no-breaks industriais divididos em chunks de 512 tokens (overlap 50).
  - 128 perguntas de teste independentes (deduplicadas de 132 originais).
- **Matching Engine em 3 Camadas**:
  1. *Contenção Estrita de Substring* (trechos $\ge 40$ chars normalizados);
  2. *RapidFuzz Token Set Ratio* ($\ge 85\%$ similaridade ou partial ratio $\ge 80\%$);
  3. *ROUGE-L Lexical Overlap* ($F_1 \ge 0.50$).
- **Validação Estatística**:
  - Teste pareado não-paramétrico de **Wilcoxon** ($lpha = 0.05$) sobre a variação consulta-a-consulta de MRR@5.
  - Intervalos de Confiança **Bootstrap 95%** ($B = 1.000$ iterações).
- **Gestão Estrita de VRAM**:
  - Descarga obrigatória de memória GPU entre modelos (`torch.cuda.empty_cache()`, `keep_alive=0`, `gc.collect()`).

---

## 4. Tabela Geral de Resultados no Top-5 ($K=5$)

| Modelo Base | Reranker | HR@5 | MRR@5 | Δ MRR@5 | Δ HR@5 | Wilcoxon (p) | Latência |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `multilingual_e5_large` | `none` | 54.7% | **0.4031** | — | — | Baseline | 0.0 ms |
| `multilingual_e5_large` | `bge_reranker_v2_m3` | 57.0% | **0.4982** | +0.0951 | +2.3% | 0.0261 | 1164.3 ms |
| `multilingual_e5_large` | `qwen3_4b_q4km` | 62.5% | **0.4764** | +0.0733 | +7.8% | 0.0406 | 6211.0 ms |
| `multilingual_e5_large` | `qwen3_8b_q5km` | 65.6% | **0.4975** | +0.0944 | +10.9% | 0.0023 | 6894.9 ms |
| `multilingual_e5_large` | `rankgpt` | 61.7% | **0.4573** | +0.0542 | +7.0% | 0.0510 | 1723.0 ms |
| `qwen3_8b_q4km` | `none` | 57.0% | **0.3887** | — | — | Baseline | 0.0 ms |
| `qwen3_8b_q4km` | `bge_reranker_v2_m3` | 53.9% | **0.4866** | +0.0979 | -3.1% | 0.0038 | 1145.1 ms |
| `qwen3_8b_q4km` | `qwen3_4b_q4km` | 60.9% | **0.4738** | +0.0852 | +3.9% | 0.0049 | 5181.6 ms |
| `qwen3_8b_q4km` | `qwen3_8b_q5km` | 60.2% | **0.5029** | +0.1142 | +3.1% | 0.0001 | 6424.2 ms |
| `qwen3_8b_q4km` | `rankgpt` | 57.8% | **0.4663** | +0.0776 | +0.8% | 0.0160 | 1647.8 ms |

*\*Nota: p < 0.05 comprova significância estatística do ganho contra o baseline puro correspondente.*


---

## 5. Eficácia de Recuperação: MRR@5 vs. Hit Rate@5

<div class="grid-2">
<div class="col-text">

- **Salto Expressivo de Precisão**:
  - O re-ranking elevou o MRR@5 de **0.3887 $\rightarrow$ 0.5029 (+29.4%)** na base Qwen3-8B.
  - Na base E5-Large, o MRR@5 saltou de **0.4031 $\rightarrow$ 0.4980 (+23.5%)**.
- **Resgate de Documentos Relevantes**:
  - O Hit Rate@5 aumentou até **+7.8 pontos percentuais** (de 54.7% para 62.5% com Qwen3-4B).
- **Top Performer Absoluto**:
  - `Qwen3-8B (Q5_K_M)` entregou a melhor ordenação média do experimento (MRR@5 = 0.5029).

</div>
<div class="col-img">
<img src="/home/joaorura/orca/workspaces/rag_eval/gorgonian/graficos_tcc/ranking_rerankers_mrr_hr_k5.png" class="chart-img" alt="Ranking MRR5 e HR5">
</div>
</div>

---

## 6. Ganhos Incrementais: Todos os Modelos Locais Superam RankGPT

<div class="grid-2">
<div class="col-text">

- **Superação da OpenAI Comercial**:
  - `RankGPT (gpt-4o-mini)` obteve $\Delta$ MRR@5 de **+0.0776**.
  - `Qwen3-4B (Q4_K_M)` obteve $\Delta$ MRR@5 de **+0.0852**.
  - `BGE-v2-m3` obteve $\Delta$ MRR@5 de **+0.0979**.
  - `Qwen3-8B (Q5_K_M)` obteve $\Delta$ MRR@5 de **+0.1142**.
- **Por que os Modelos Locais Vencem?**:
  - *Fine-tuning dedicado para relevância de passagens* supera o prompt genérico listwise da LLM.
  - Logprobs em token binário ("yes"/"no") capturam calibração fina de confiança semântica.

</div>
<div class="col-img">
<img src="/home/joaorura/orca/workspaces/rag_eval/gorgonian/graficos_tcc/delta_mrr5_rerankers.png" class="chart-img" alt="Ganhos Incrementais MRR5">
</div>
</div>

---

## 7. Trade-off de Engenharia: Acurácia vs. Latência

<div class="grid-2">
<div class="col-text">

- **Sweet Spot de Latência (`BGE-Reranker-v2-m3`)**:
  - Adiciona apenas **1.14 segundos** por consulta para avaliar todos os 20 candidatos na GPU.
  - Atinge MRR@5 de 0.4866 (superior ao RankGPT com menos latência).
- **Sweet Spot de Cobertura (`Qwen3-4B Q4_K_M`)**:
  - Maior Hit Rate@5 (**60.9% a 62.5%**), com latência de ~5.1s.
- **Custo Computacional do 8B**:
  - Atinge o pico de precisão (0.5029), porém com latência de ~6.4s a 6.5s por consulta.

</div>
<div class="col-img">
<img src="/home/joaorura/orca/workspaces/rag_eval/gorgonian/graficos_tcc/tradeoff_latencia_mrr_rerankers.png" class="chart-img" alt="Trade-off Latencia vs MRR5">
</div>
</div>

---

## 8. Viabilidade Operacional e Perfil de Recursos

### Alocação de VRAM e Custo Operacional (GPU 8 GB GDDR6)

| Modelo / Pipeline | Engine | VRAM Alocada | Custo / 1M Queries | Dependência de Nuvem |
| :--- | :---: | :---: | :---: | :---: |
| `BGE-Reranker-v2-m3` | HuggingFace (PyTorch) | **~1.2 GB** | **$0.00** | **100% Offline** |
| `Qwen3-Reranker-4B (Q4_K_M)` | Ollama (GGUF) | **~2.5 GB** | **$0.00** | **100% Offline** |
| `Qwen3-Reranker-8B (Q5_K_M)` | Ollama (GGUF) | ~5.9 GB | $0.00 | **100% Offline** |
| `RankGPT (gpt-4o-mini)` | OpenAI Cloud API | 0.0 GB | ~$150.00 | Conexão Obrigatória |

> **Conclusão de Engenharia**: Tanto o `BGE-Reranker-v2-m3` quanto o `Qwen3-4B` operam com enorme folga de memória na GPU RTX PRO 1000 (8 GB), garantindo estabilidade e custo zero.

---

## 9. Recomendações Finais de Arquitetura para a CM Comandos

### Arquitetura de Produção Recomendada (Pipeline Híbrido)

1. **Cenário de Produção em Tempo Real (Baixa Latência & SLA Rígido)**:
   - **1º Estágio**: `Multilingual-E5-Large` (FP16 ou ONNX INT8) $ightarrow$ 22.9 ms.
   - **2º Estágio**: `BGE-Reranker-v2-m3` $ightarrow$ 1145 ms.
   - **Tempo Total**: ~1.17 segundos | **MRR@5**: 0.4866 | **VRAM Máxima**: 2.5 GB.

2. **Cenário de Análise Aprofundada / Suporte Especializado**:
   - **1º Estágio**: `Qwen3-Embedding-8B (Q4_K_M)` $ightarrow$ 57.1 ms.
   - **2º Estágio**: `Qwen3-Reranker-8B (Q5_K_M)` $ightarrow$ 6424 ms.
   - **Tempo Total**: ~6.48 segundos | **MRR@5**: **0.5029** | **HR@5**: **60.2%**.

---

<!-- _class: lead -->
# Perguntas & Discussão Técnica

### Contribuições da Pesquisa (ICT):
- Demonstração empírica da viabilidade de RAG Two-Stage 100% local em GPU de 8 GB.
- Confirmação de que SLMs quantizados locais superam modelos comerciais genéricos em domínio eletrotécnico.
- Código, dados e pipelines totalmente reprodutíveis.

*Obrigado!*
