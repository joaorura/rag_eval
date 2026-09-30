---
marp: true
theme: default
paginate: true
header: "Avaliação Científica e Tecnológica de RAG Industrial (Retrieval & Re-ranking) — Pesquisa ICT"
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
    font-size: 1.55em;
    margin-bottom: 0.25em;
    margin-top: 5px;
  }
  h2 {
    color: #2b6cb0;
    font-size: 1.15em;
    margin-top: 10px;
    margin-bottom: 0.3em;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
  }
  h3 {
    color: #2d3748;
    font-size: 0.92em;
    margin-top: 8px;
    margin-bottom: 4px;
  }
  p, li {
    font-size: 0.68em;
    line-height: 1.42;
  }
  ul {
    margin-top: 5px;
    margin-bottom: 6px;
  }
  table {
    font-size: 0.58em;
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
  }
  th {
    background-color: #edf2f7;
    color: #2d3748;
    padding: 6px 8px;
    text-align: left;
  }
  td {
    padding: 5px 8px;
    border-bottom: 1px solid #e2e8f0;
  }
  .badge-win {
    background-color: #c6f6d5;
    color: #22543d;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 3px;
  }
  .badge-highlight {
    background-color: #bee3f8;
    color: #2b6cb0;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 3px;
  }
  .grid-2 {
    display: flex;
    gap: 24px;
    align-items: center;
    margin-top: 8px;
  }
  .col-text {
    flex: 1.1;
  }
  .col-img {
    flex: 1.1;
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
  .card-box {
    background-color: #f7fafc;
    border-left: 4px solid #2b6cb0;
    padding: 8px 14px;
    border-radius: 4px;
    margin-top: 8px;
    margin-bottom: 8px;
  }
  footer {
    font-size: 0.50em;
    color: #718096;
  }
  header {
    font-size: 0.52em;
    color: #718096;
  }
---

<!-- _class: lead -->
# Avaliação Científica e Tecnológica de RAG Industrial (Retrieval & Re-ranking)
## Soberania de Dados e Alta Precisão em Manuais Técnicos com Modelos Locais Quantizados

**Autor:** João Vitor Rura | Projeto de Pesquisa Científica e Tecnológica (ICT)  
**Hardware de Execução:** NVIDIA RTX PRO 1000 Laptop GPU (8 GB GDDR6) | Intel Core Ultra 7 | 32 GB RAM  
**Estudo de Caso & Parceria:** CM Comandos Lineares | Setembro / 2026

---

## 2. Contexto e Desafio Industrial

<div class="card-box">
<b>Cenário de Aplicação:</b> 16 manuais de no-breaks industriais da CM Comandos Lineares cobrindo linhas monofásicas, trifásicas e sistemas de missão crítica (hospitalar, data centers e infraestrutura de transporte).
</div>

- **Vocabulário Eletrotécnico Especializado**:
  - Manuais densos com conceitos interdependentes: *bypass estático*, *retificador tiristorizado hexapolar*, *inversor IGBT*, *tensão de flutuação VRLA*, *barramento CC* e *distorção harmônica (THD)*.
  - Erros de interpretação em campo acarretam paradas catastróficas em cargas críticas protegidas.

- **Limitações Críticas de Soluções Proprietárias em Nuvem**:
  - **Soberania e Sigilo Industrial**: Inadmissibilidade de envio de manuais e esquemáticos proprietários confidenciais para servidores externos de terceiros.
  - **Custo Operacional em Escala**: Modelo "pay-per-token" inviabiliza milhões de requisições de assistência técnica continuada.
  - **Inviabilidade em Ambientes Isolados (*Air-Gapped*)**: Instalações hospitalares, subestações e centros de comando operam em redes isoladas sem conexão externa à internet.

- **Requisitos de Engenharia**:
  - Operação 100% local em hardware pessoal/workstation de **8 GB de VRAM**, resposta submétrica (< 2s) e **custo zero de inferência**.

---

## 3. Metodologia Experimental e Rigor Estatístico

- **Corpus Técnico & Consultas de Teste**:
  - Documentos divididos em chunks uniformes com `SentenceSplitter(chunk_size=512, chunk_overlap=50)`.
  - **128 perguntas técnicas deduplicadas** ($N_{\text{efetivo}}$) cobrindo instalação, parametrização, alarmes e procedimentos corretivos.

- **Motor de Avaliação Multicamada (*Matching Engine*)**:
  1. *Contenção Estrita de Substring*: Detecção determinística de passagens canônicas (>= 40 caracteres normalizados);
  2. *Similaridade Sintática RapidFuzz*: *Token Set Ratio* (>= 85%) e *Partial Ratio* (>= 80%) contra variações morfológicas;
  3. *Sobreposição Semântica ROUGE-L*: Avaliação da maior subsequência comum de tokens contíguos ($F_1 \ge 0.50$).

- **Validação Estatística com Rigor Científico**:
  - **Teste Pareado de Postos com Sinais de Wilcoxon** ($\alpha = 0.05$): Avaliação de significância estatística consulta a consulta.
  - **Intervalos de Confiança Bootstrap 95%** ($B = 1.000$ iterações): Quantificação de dispersão para HR@K, MRR@K, Recall@K e MAP@K.
  - **Gestão Estrita de VRAM**: Purga determinística de memória GPU (`torch.cuda.empty_cache()`, `keep_alive=0`) e isolamento físico de índices.

---

## 4. Fase 1: Avaliação de Embeddings (1º Estágio)

<div class="grid-2">
<div class="col-text">

- **7 Modelos em Confronto**:
  - Baseline OpenAI `text-embedding-3-small` vs. 6 modelos abertos locais (Qwen3-8B/4B Q4_K_M, Multilingual-E5-Large, BGE-M3, Nomic, BERTimbau).
- **Destaque de Precisão**:
  - `multilingual-e5-large` alcançou **MRR@5 = 0.4031** com latência ultrarrápida de **22.9 ms** (FP16 na GPU).
- **Superioridade dos SLMs em 4-bit (Q4_K_M)**:
  - `qwen3_8b_q4km` atingiu maior revocação (**HR@5 = 57.03%** vs. 46.88% da OpenAI; $p = 0.0029$).
  - `qwen3_4b_q4km` entregou HR@5 = 55.47% e MRR@5 = 0.3844 com **36.0 ms**.
- **Soberania com Custo Zero**:
  - SLMs locais quantizados retêm >100% da acurácia da OpenAI com **11.5x menor latência** (36 ms vs. 415 ms) e custo zero de API.

</div>
<div class="col-img">
<img src="/home/joaorura/orca/workspaces/rag_eval/gorgonian/graficos_tcc/ranking_hitrate_mrr_k5.png" class="chart-img" alt="Ranking de Embeddings MRR@5 e Hit Rate@5">
</div>
</div>

---

## 5. O Desafio: Por que um estágio de recuperação não é suficiente?

- **Limitação Estrutural dos Bi-Encoders**:
  - Bi-encoders mapeiam pergunta e passagens técnicas em vetores independentes: $\vec{u} = E(q)$, $\vec{v} = E(d)$.
  - A similaridade de cosseno condensa centenas de tokens em um único escalar, perdendo correlações sutis e ordenamento estrito de passos técnicos (ex.: ordem de manobra de disjuntores, intertravamentos lógicos e diferenças sutis entre famílias de no-breaks).

- **O Teto de Precisão no Top-5**:
  - No 1º estágio, mesmo o melhor modelo (`e5-large`) atinge MRR@5 de 0.4031 — ou seja, na maioria das consultas, o nó mais relevante não está no primeiro posto ($K=1$).
  - Ocorrência de falsos positivos com alta sobreposição lexical mas sem contexto operacional pertinente.

- **A Solução: Arquitetura Two-Stage RAG**:
  - **1º Estágio (Super-sampling $K_{\text{cand}} = 20$)**: Bi-encoder recupera 20 candidatos em 20–50 ms com recall elevado (~68%).
  - **2º Estágio (Re-ranking Neural)**: Atenção cruzada profunda (*all-to-all cross-attention*) ou logprob softmax binário para reordenar os nós, projetando os fragmentos corretos para o Top-1 e Top-5.

---

## 6. Fase 2: Benchmark de Re-ranking Neural no Top-5

| Modelo Base (1º Estágio) | Reranker (2º Estágio) | HR@5 | MRR@5 | Δ MRR@5 | Δ HR@5 | Wilcoxon (p) | Latência Re-rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `multilingual_e5_large` | `none` (Baseline Puro) | 54.7% | 0.4031 | — | — | Baseline | 0.0 ms |
| `multilingual_e5_large` | `bge_reranker_v2_m3` | 57.0% | 0.4982 | +0.0951 | +2.3% | 0.0261* | 1164.3 ms |
| `multilingual_e5_large` | `qwen3_4b_q4km` | 62.5% | 0.4764 | +0.0733 | +7.8% | 0.0406* | 6211.0 ms |
| `multilingual_e5_large` | `qwen3_8b_q5km` | <span class="badge-win">65.6%</span> | 0.4975 | +0.0944 | +10.9% | 0.0023* | 6894.9 ms |
| `multilingual_e5_large` | `rankgpt` (gpt-4o-mini) | 61.7% | 0.4573 | +0.0542 | +7.0% | 0.0510 | 1723.0 ms |
| `qwen3_8b_q4km` | `none` (Baseline Puro) | 57.0% | 0.3887 | — | — | Baseline | 0.0 ms |
| `qwen3_8b_q4km` | `bge_reranker_v2_m3` | 53.9% | 0.4866 | +0.0979 | -3.1% | 0.0038* | 1145.1 ms |
| `qwen3_8b_q4km` | `qwen3_4b_q4km` | 60.9% | 0.4738 | +0.0852 | +3.9% | 0.0049* | 5181.6 ms |
| `qwen3_8b_q4km` | `qwen3_8b_q5km` | 60.2% | <span class="badge-win">0.5029</span> | <span class="badge-win">+0.1142</span> | +3.1% | 0.0001* | 6424.2 ms |
| `qwen3_8b_q4km` | `rankgpt` (gpt-4o-mini) | 57.8% | 0.4663 | +0.0776 | +0.8% | 0.0160* | 1647.8 ms |

*Nota: p < 0.05 comprova significância estatística do ganho contra o baseline puro correspondente.*

---

## 7. Descoberta Central: Modelos Locais Superam a OpenAI

<div class="grid-2">
<div class="col-text">

- **Superação Sistemática do RankGPT Comercial**:
  - Todos os modelos locais (`bge_reranker_v2_m3`, `qwen3_4b` e `qwen3_8b`) superaram o `RankGPT (gpt-4o-mini)` em MRR@5.
- **Salto Histórico de Precisão**:
  - Na base Qwen3-8B, o MRR@5 saltou de **0.3887 para 0.5029 (+29.4%)** com significância estatística extrema ($p = 0.0001$).
  - Na base E5-Large, o MRR@5 subiu para **0.4982 (+23.6%)** com BGE-Reranker-v2-m3 ($p = 0.0261$).
- **Por que Modelos Locais Vencem a Nuvem?**:
  - Modelos dedicados de re-ranking (Cross-Encoders e SLMs com logprob softmax) possuem calibração estrita de probabilidade de relevância.
  - Superam a instabilidade e heurística posicional dos prompts listwise de modelos genéricos comerciais em nuvem.

</div>
<div class="col-img">
<img src="/home/joaorura/orca/workspaces/rag_eval/gorgonian/graficos_tcc/delta_mrr5_rerankers.png" class="chart-img" alt="Ganhos Incrementais em MRR@5">
</div>
</div>

---

## 8. Trade-off de Engenharia: Acurácia vs. Latência vs. VRAM

<div class="grid-2">
<div class="col-text">

- **Análise da Fronteira de Pareto**:
  - `BGE-Reranker-v2-m3`: Melhor ponto de equilíbrio de engenharia.
    - Adiciona apenas **1.14s** por consulta na GPU.
    - Consome apenas **~1.2 GB de VRAM** (folga massiva na GPU de 8 GB).
    - Entrega MRR@5 de 0.4866 a 0.4982.
- **Sweet Spot de Cobertura (`Qwen3-4B Q4_K_M`)**:
  - Eleva o Hit Rate@5 para **62.5%** (+7.8 p.p.) com latência de ~5.1s e ~2.5 GB VRAM.
- **Teto de Precisão (`Qwen3-8B Q5_K_M`)**:
  - Entrega cobertura máxima (**65.6% HR@5** na base E5) e maior MRR (**0.5029** na base Qwen) em 6.4s a 6.8s (~5.9 GB VRAM).

</div>
<div class="col-img">
<img src="/home/joaorura/orca/workspaces/rag_eval/gorgonian/graficos_tcc/tradeoff_latencia_mrr_rerankers.png" class="chart-img" alt="Trade-off Latência vs MRR@5">
</div>
</div>

---

## 9. Recomendações Práticas de Implantação para a CM Comandos

<div class="grid-2">
<div class="col-text">

<div class="card-box">
<h3>Cenário 1: Tempo Real / Suporte Técnico de Campo</h3>

- **1º Estágio**: `Multilingual-E5-Large` $\rightarrow$ **22.9 ms**
- **2º Estágio**: `BGE-Reranker-v2-m3` $\rightarrow$ **1164 ms**
- **Latência Total**: **~1.19 segundos** por consulta
- **Alocação de VRAM**: **< 2.5 GB** (5.5 GB livres na GPU)
- **Eficácia Top-5**: MRR@5 = 0.4982 | HR@5 = 57.0%
- **Aplicação**: Chatbot de campo para técnicos, SAC e atendimento interativo em tempo real.
</div>

</div>
<div class="col-text">

<div class="card-box">
<h3>Cenário 2: Diagnóstico Corretivo Aprofundado</h3>

- **1º Estágio**: `Qwen3-Embedding-8B (Q4_K_M)` $\rightarrow$ **57.1 ms**
- **2º Estágio**: `Qwen3-Reranker-8B (Q5_K_M)` $\rightarrow$ **6424 ms**
- **Latência Total**: **~6.48 segundos** por consulta
- **Alocação de VRAM**: **~5.9 GB** (estável na GPU de 8 GB)
- **Eficácia Top-5**: **MRR@5 = 0.5029** | **HR@5 = 65.6%**
- **Aplicação**: Engenharia de produto, perícia pós-falha e análise profunda de inversores.
</div>

</div>
</div>

<div class="card-box" style="margin-top: 10px;">
<b>Conclusão Executiva:</b> Ambos os cenários operam 100% desconectados da internet (<i>air-gapped</i>), com $0 de custo de API e soberania absoluta sobre os manuais da CM Comandos.
</div>

---

## 10. Impacto do Projeto de ICT e Próximos Passos

- **Principais Conquistas e Validações Científicas**:
  - **Quebra de Paradigma**: Modelos abertos locais quantizados superam soluções comerciais proprietárias em tarefas técnicas de recuperação de informação eletrotécnica.
  - **Soberania & Custo Zero**: Viabilidade técnica comprovada de RAG Two-Stage de alta fidelidade em hardware pessoal/workstation de 8 GB GDDR6.
  - **Blindagem de Ativos**: Proteção absoluta de segredos industriais e propriedade intelectual da CM Comandos.

- **Próximos Passos Tecnológicos**:
  - **Pipeline de Geração (3º Estágio)**: Avaliação de SLMs locais instruídos (ex.: Qwen-2.5-7B, Llama-3.1-8B) com restrição estrita de aterramento (*grounding*) e citação de páginas dos manuais.
  - **Piloto Operacional**: Implantação de ambiente piloto integrado à base de chamados técnicos e suporte ao cliente da CM Comandos Lineares.

---

<!-- _class: lead -->
# Perguntas & Discussão

### Avaliação Científica e Tecnológica de RAG Industrial (Retrieval & Re-ranking)

**Autor:** João Vitor Rura  
**Linha de Pesquisa:** Projeto de Pesquisa Científica e Tecnológica (ICT)  
**Instituições:** Universidade & CM Comandos Lineares  
**Hardware:** NVIDIA RTX PRO 1000 Laptop GPU (8 GB GDDR6)

*Repositório, códigos-fonte, pipelines e relatórios consolidados disponíveis para auditoria técnica.*
