"""Gera a apresentação executiva completa para reuniões sobre toda a pesquisa de RAG Industrial (Embeddings e Re-ranking).

Enquadramento: Projeto de Pesquisa Científica e Tecnológica (ICT).
Hardware: NVIDIA RTX PRO 1000 Laptop GPU (8 GB GDDR6) | Parceria: CM Comandos Lineares.

Gera os seguintes artefatos:
1. docs/apresentacao_pesquisa_reuniao.md (Marp Markdown 16:9 estilizado)
2. docs/apresentacao_pesquisa_reuniao.html (Compilado via @marp-team/marp-cli)
3. docs/apresentacao_pesquisa_reuniao.pdf (Compilado via @marp-team/marp-cli --pdf)
4. docs/apresentacao_pesquisa_reuniao.pptx (Apresentação PowerPoint editável via python-pptx)
5. ./apresentacao_pesquisa_reuniao.pdf (Cópia para a raiz do repositório)
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor


def to_base64_img(img_path: str) -> str:
    """Converte arquivo PNG em data URI base64 para incorporar diretamente no HTML e PDF."""
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    return img_path


def build_marp_markdown(base_dir: str) -> str:
    img_ranking_emb = to_base64_img(os.path.join(base_dir, "graficos_tcc", "ranking_hitrate_mrr_k5.png"))
    img_delta_rerank = to_base64_img(os.path.join(base_dir, "graficos_tcc", "delta_mrr5_rerankers.png"))
    img_tradeoff_rerank = to_base64_img(os.path.join(base_dir, "graficos_tcc", "tradeoff_latencia_mrr_rerankers.png"))

    marp_content = f"""---
marp: true
theme: default
paginate: true
header: "RAG Industrial em Manuais Técnicos — Pesquisa ICT"
footer: "CM Comandos Lineares | Pesquisa Científica e Tecnológica (ICT)"
style: |
  section {{
    font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 30px 45px;
    background-color: #ffffff;
    color: #1a202c;
  }}
  h1 {{
    color: #1a365d;
    font-size: 1.55em;
    margin-bottom: 0.25em;
    margin-top: 5px;
  }}
  h2 {{
    color: #2b6cb0;
    font-size: 1.15em;
    margin-top: 10px;
    margin-bottom: 0.3em;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
  }}
  h3 {{
    color: #2d3748;
    font-size: 0.92em;
    margin-top: 8px;
    margin-bottom: 4px;
  }}
  p, li {{
    font-size: 0.68em;
    line-height: 1.42;
  }}
  ul {{
    margin-top: 5px;
    margin-bottom: 6px;
  }}
  table {{
    font-size: 0.58em;
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
  }}
  th {{
    background-color: #edf2f7;
    color: #2d3748;
    padding: 6px 8px;
    text-align: left;
  }}
  td {{
    padding: 5px 8px;
    border-bottom: 1px solid #e2e8f0;
  }}
  .badge-win {{
    background-color: #c6f6d5;
    color: #22543d;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 3px;
  }}
  .badge-highlight {{
    background-color: #bee3f8;
    color: #2b6cb0;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 3px;
  }}
  .grid-2 {{
    display: flex;
    gap: 24px;
    align-items: center;
    margin-top: 8px;
  }}
  .col-text {{
    flex: 1.1;
  }}
  .col-img {{
    flex: 1.1;
    text-align: center;
  }}
  .chart-img {{
    width: 100%;
    max-height: 420px;
    object-fit: contain;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }}
  .card-box {{
    background-color: #f7fafc;
    border-left: 4px solid #2b6cb0;
    padding: 8px 14px;
    border-radius: 4px;
    margin-top: 8px;
    margin-bottom: 8px;
  }}
  footer {{
    font-size: 0.50em;
    color: #718096;
  }}
  header {{
    font-size: 0.46em;
    color: #718096;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 600;
    white-space: nowrap;
  }}
---

<!-- _class: lead -->
# Avaliação Científica e Tecnológica de RAG Industrial (Retrieval & Re-ranking)
## Soberania de Dados e Alta Precisão em Manuais Técnicos com Modelos Locais Quantizados

**Autor:** João Messias Lima Pereira | Projeto de Pesquisa Científica e Tecnológica (ICT)  
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
  - **128 perguntas técnicas deduplicadas** ($N_{{\\text{{efetivo}}}}$) cobrindo instalação, parametrização, alarmes e procedimentos corretivos.

- **Motor de Avaliação Multicamada (*Matching Engine*)**:
  1. *Contenção Estrita de Substring*: Detecção determinística de passagens canônicas (>= 40 caracteres normalizados);
  2. *Similaridade Sintática RapidFuzz*: *Token Set Ratio* (>= 85%) e *Partial Ratio* (>= 80%) contra variações morfológicas;
  3. *Sobreposição Semântica ROUGE-L*: Avaliação da maior subsequência comum de tokens contíguos ($F_1 \\ge 0.50$).

- **Validação Estatística com Rigor Científico**:
  - **Teste Pareado de Postos com Sinais de Wilcoxon** ($\\alpha = 0.05$): Avaliação de significância estatística consulta a consulta.
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
<img src="{img_ranking_emb}" class="chart-img" alt="Ranking de Embeddings MRR@5 e Hit Rate@5">
</div>
</div>

---

## 5. O Desafio: Por que um estágio de recuperação não é suficiente?

- **Limitação Estrutural dos Bi-Encoders**:
  - Bi-encoders mapeiam pergunta e passagens técnicas em vetores independentes: $\\vec{{u}} = E(q)$, $\\vec{{v}} = E(d)$.
  - A similaridade de cosseno condensa centenas de tokens em um único escalar, perdendo correlações sutis e ordenamento estrito de passos técnicos (ex.: ordem de manobra de disjuntores, intertravamentos lógicos e diferenças sutis entre famílias de no-breaks).

- **O Teto de Precisão no Top-5**:
  - No 1º estágio, mesmo o melhor modelo (`e5-large`) atinge MRR@5 de 0.4031 — ou seja, na maioria das consultas, o nó mais relevante não está no primeiro posto ($K=1$).
  - Ocorrência de falsos positivos com alta sobreposição lexical mas sem contexto operacional pertinente.

- **A Solução: Arquitetura Two-Stage RAG**:
  - **1º Estágio (Super-sampling $K_{{\\text{{cand}}}} = 20$)**: Bi-encoder recupera 20 candidatos em 20–50 ms com recall elevado (~68%).
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
<img src="{img_delta_rerank}" class="chart-img" alt="Ganhos Incrementais em MRR@5">
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
<img src="{img_tradeoff_rerank}" class="chart-img" alt="Trade-off Latência vs MRR@5">
</div>
</div>

---

## 9. Recomendação de Produção: Arquitetura 100% em CPU (Sem GPU Dedicada)

<div class="grid-2">
<div class="col-text">

<div class="card-box">
<h3>Contexto da Infraestrutura de Produção</h3>

- **Servidor Corporativo Exclusivo em CPU**:
  - O servidor de produção da CM Comandos opera **exclusivamente em CPU** (sem placa gráfica aceleradora).
  - Soberania total: proibição de tráfego de manuais confidenciais para servidores em nuvem.
- **Solução Campeã Homologada**:
  - 1º Estágio com **`Multilingual-E5-Large (ONNX INT8)`**.
  - Acelerado nativamente via instruções vetoriais **AVX-512 / VNNI** da CPU.
- **Custo e Segurança Operacional**:
  - **Custo zero de API** e blindagem total de segredos industriais em rede local (*air-gapped*).
</div>

</div>
<div class="col-text">

<div class="card-box" style="border-left-color: #38a169;">
<h3>Métricas Comprovadas na CPU (Servidor CM Comandos)</h3>

- **Latência Ultrarrápida na CPU**:
  - Apenas **107 ms por consulta** na CPU pura.
  - **3,9x mais rápido** que os 415 ms da API da OpenAI em nuvem!
- **Consumo Mínimo de Memória**:
  - **< 1.5 GB de RAM** (perfeitamente estável e viável em qualquer VM modesta).
- **Precisão Superior ao Baseline Comercial**:
  - **MRR@5 de 0.4031** (já superior aos 0.3840 da OpenAI `text-embedding-3-small`).
- **Prontidão para Atendimento em Tempo Real**:
  - Resposta instantânea (< 200 ms) para técnicos de campo e suporte.
</div>

</div>
</div>

<div class="card-box" style="margin-top: 10px;">
<b>Veredito de Engenharia:</b> O <code>Multilingual-E5-Large (ONNX INT8)</code> é a recomendação definitiva para o servidor de produção da CM Comandos: velocidade de 107 ms e precisão superior à nuvem, operando 100% em CPU sem custos de acelerador ou API.
</div>

---

## 10. Estratégias Complementares e Limites de Re-ranking na CPU

<div class="grid-2">
<div class="col-text">

<div class="card-box" style="border-left-color: #e53e3e;">
<h3>Limites na CPU: Por que SLMs Foram Descartados</h3>

- **Inviabilidade de SLMs Generativos na CPU**:
  - Modelos generativos de 4B e 8B (`Qwen3-Reranker`) exigem de **20s a 40s por consulta** na CPU pura.
  - Latência proibitiva que inviabiliza completamente o uso em atendimento interativo e suporte em tempo real.
- **Reserva de Capacidade Computacional**:
  - Descartar SLMs na CPU evita enfileiramento e mantém o servidor responsivo para múltiplos técnicos simultâneos.
</div>

</div>
<div class="col-text">

<div class="card-box" style="border-left-color: #3182ce;">
<h3>Busca Híbrida Leve (Altamente Recomendada) 🥇</h3>

- **Fusão Densa + Léxica via RRF**:
  - `Multilingual-E5-Large (ONNX INT8)` + `BM25` via *Reciprocal Rank Fusion*.
  - Captura códigos de falha (ex.: "F07", "E-12") e números de peças com latência adicional de **apenas ~5 ms na CPU** (~112 ms total).
</div>

<div class="card-box" style="border-left-color: #805ad5; margin-top: 6px;">
<h3>Opção com Re-ranking Neural Leve (Se Necessário) ⚡</h3>

- **Cross-Encoder Compacto em CPU**:
  - `BGE-Reranker-v2-m3 (ONNX INT8)` restrito a um pool pequeno ($K_{{\\text{{cand}}}} = 5$ nós).
  - Atinge latência de **~1.5s a 1.8s na CPU** mantendo consumo < 2.5 GB RAM.
</div>

</div>
</div>

<div class="card-box" style="margin-top: 10px;">
<b>Recomendação Final:</b> A Busca Híbrida (E5-Large ONNX + BM25) fornece o melhor trade-off na CPU: máxima precisão léxica e semântica com latência de apenas ~112 ms, sem gargalos computacionais.
</div>

---

## 11. Impacto do Projeto de ICT e Próximos Passos

- **Principais Conquistas e Validações Científicas**:
  - **Quebra de Paradigma Tecnológico**: Comprovação com rigor estatístico (Wilcoxon $p < 0.05$) de que modelos abertos locais superam soluções comerciais proprietárias em documentações eletrotécnicas.
  - **Viabilidade Comprovada em CPU Pura**: Homologação do `Multilingual-E5-Large (ONNX INT8)` operando em apenas **107 ms na CPU** (< 1.5 GB RAM), eliminando a dependência de GPUs caras.
  - **Soberania Integral & Custo Zero**: Blindagem de segredos industriais e manuais confidenciais de no-breaks da CM Comandos sem dependência de nuvem.

- **Próximos Passos Tecnológicos (Foco em CPU)**:
  - **Validação e Testes de Carga em CPU Corporativa**: Testes de concorrência simultânea do pipeline E5-Large ONNX INT8 + BM25 na infraestrutura de servidores da CM Comandos.
  - **Pipeline de Geração Leve (3º Estágio)**: Avaliação de modelos geradores compactos quantizados eficientes em CPU para síntese de respostas com citação exata de páginas dos manuais.
  - **Piloto Operacional Integrado**: Disponibilização da ferramenta para homologação prática com técnicos de campo e equipe de assistência da CM Comandos Lineares.

---

<!-- _class: lead -->
# Perguntas & Discussão

### Avaliação Científica e Tecnológica de RAG Industrial (Retrieval & Re-ranking)

**Autor:** João Messias Lima Pereira  
**Linha de Pesquisa:** Projeto de Pesquisa Científica e Tecnológica (ICT)  
**Instituições:** Universidade & CM Comandos Lineares  
**Hardware de Referência:** Intel Core Ultra 7 (CPU) & NVIDIA RTX PRO 1000 (8 GB GDDR6)

*Repositório, códigos-fonte, pipelines e relatórios consolidados disponíveis para auditoria técnica.*
"""
    return marp_content


def build_pptx_presentation(output_path: str, base_dir: str) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    color_primary = RGBColor(26, 54, 93)      # #1a365d
    color_tech_blue = RGBColor(43, 108, 176)  # #2b6cb0
    color_dark_text = RGBColor(45, 55, 72)    # #2d3748
    color_muted = RGBColor(113, 128, 150)     # #718096
    color_win_green = RGBColor(34, 139, 34)   # Forest green
    color_card_bg = RGBColor(247, 250, 252)   # #f7fafc
    color_border = RGBColor(226, 232, 240)    # #e2e8f0

    def add_header(slide, title_text: str, category: str = "RAG INDUSTRIAL EM MANUAIS TÉCNICOS — PESQUISA ICT"):
        c_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.35))
        tf_c = c_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = color_tech_blue

        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.733), Inches(0.75))
        tf = t_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(21)
        p.font.bold = True
        p.font.color.rgb = color_primary

    def add_bullet(tf, title: str, body: str, size: int = 12, space_before: int = 8):
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = title + (": " if title else "")
        p.font.bold = True
        p.font.size = Pt(size)
        p.font.color.rgb = color_tech_blue
        p.space_before = Pt(space_before)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.color.rgb = color_dark_text

    def add_card_box(slide, left, top, width, height, bg_rgb=color_card_bg, border_rgb=color_border):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_rgb
        shape.line.color.rgb = border_rgb
        shape.line.width = Pt(1.5)
        return shape

    # =========================================================================
    # SLIDE 1: Capa Executiva
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    box1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.2))
    tf1 = box1.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "Avaliação Científica e Tecnológica de RAG Industrial\n(Retrieval & Re-ranking)"
    p1.font.size = Pt(30)
    p1.font.bold = True
    p1.font.color.rgb = color_primary

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Soberania de Dados e Alta Precisão em Manuais Técnicos com Modelos Locais Quantizados"
    p1_sub.font.size = Pt(17)
    p1_sub.font.bold = True
    p1_sub.font.color.rgb = color_tech_blue
    p1_sub.space_before = Pt(16)

    p1_meta = tf1.add_paragraph()
    p1_meta.text = "Autor: João Messias Lima Pereira  |  Projeto de Pesquisa Científica e Tecnológica (ICT)\nHardware: NVIDIA RTX PRO 1000 Laptop GPU (8 GB GDDR6) & Intel Core Ultra 7 (CPU)  |  Estudo de Caso: CM Comandos Lineares  |  Setembro / 2026"
    p1_meta.font.size = Pt(12)
    p1_meta.font.color.rgb = color_muted
    p1_meta.space_before = Pt(28)

    # =========================================================================
    # SLIDE 2: Contexto e Desafio Industrial
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Contexto Operacional e Desafio Industrial")

    # Card superior: Cenário
    add_card_box(s2, Inches(0.8), Inches(1.5), Inches(11.733), Inches(0.95))
    tb_sc = s2.shapes.add_textbox(Inches(0.95), Inches(1.55), Inches(11.4), Inches(0.85))
    tf_sc = tb_sc.text_frame
    tf_sc.word_wrap = True
    p_sc = tf_sc.paragraphs[0]
    p_sc.text = "Cenário de Aplicação: "
    p_sc.font.bold = True
    p_sc.font.size = Pt(12)
    p_sc.font.color.rgb = color_tech_blue
    r_sc = p_sc.add_run()
    r_sc.text = "16 manuais de no-breaks industriais da CM Comandos Lineares cobrindo equipamentos monofásicos, trifásicos e sistemas modulares de missão crítica (hospitalar, data centers e infraestrutura de alta confiabilidade)."
    r_sc.font.bold = False
    r_sc.font.color.rgb = color_dark_text

    # Bullets de conteúdo
    tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(2.65), Inches(11.733), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    add_bullet(tf2, "Vocabulário Eletrotécnico Especializado",
               "Manuais ricos em jargão de alta densidade técnica (bypass estático, retificador tiristorizado hexapolar, inversor IGBT, tensão de flutuação VRLA, barramento CC, THD). Erros de interpretação em campo acarretam falhas críticas em cargas protegidas.",
               size=12, space_before=6)
    add_bullet(tf2, "Limitações de Soluções Proprietárias em Nuvem",
               "Inadmissibilidade de tráfego de esquemáticos confidenciais e manuais industriais para APIs de terceiros (risco de propriedade intelectual). Custo 'pay-per-token' inviabiliza milhões de requisições de assistência técnica continuada.",
               size=12, space_before=10)
    add_bullet(tf2, "Inviabilidade em Ambientes Isolados (Air-Gapped)",
               "Centros cirúrgicos hospitalares, usinas e subestações operam sob isolamento rígido de rede, demandando inferência 100% desconectada da internet.",
               size=12, space_before=10)
    add_bullet(tf2, "Requisitos Estritos de Engenharia",
               "Operação integral em hardware pessoal/workstation com 8 GB de VRAM, resposta submétrica (< 2 segundos) e custo financeiro zero de inferência.",
               size=12, space_before=10)

    # =========================================================================
    # SLIDE 3: Metodologia Experimental e Rigor Estatístico
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Metodologia Experimental e Rigor Estatístico")

    tb3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.55), Inches(11.733), Inches(5.5))
    tf3 = tb3.text_frame
    tf3.word_wrap = True

    add_bullet(tf3, "Corpus Técnico & Deduplicação Rigorosa",
               "16 manuais de no-breaks industriais divididos uniformemente via SentenceSplitter (chunk_size=512, chunk_overlap=50). Avaliação conduzida sobre 128 perguntas técnicas deduplicadas (N_efetivo) com gabaritos determinísticos de passagens canônicas.",
               size=12, space_before=4)
    add_bullet(tf3, "Motor de Avaliação Multicamada (Matching Engine)",
               "1. Contenção Estrita de Substring: Detecção determinística de passagens canônicas (>= 40 caracteres normalizados).\n"
               "2. Similaridade Sintática RapidFuzz: Token Set Ratio (>= 85%) e Partial Ratio (>= 80%) contra variações lexicais.\n"
               "3. Sobreposição Semântica ROUGE-L: Avaliação da maior subsequência comum de tokens contíguos (F1 >= 0.50).",
               size=12, space_before=10)
    add_bullet(tf3, "Inferência Estatística & Confiabilidade Científica",
               "Aplicação do teste pareado não-paramétrico de postos com sinais de Wilcoxon (alpha = 0.05) consulta por consulta para testar superioridade/equivalência. Complementado por intervalos de confiança Bootstrap de 95% (B = 1.000 iterações).",
               size=12, space_before=10)
    add_bullet(tf3, "Gestão Estrita de VRAM e Isolamento de Índices",
               "Purga determinística de memória GPU entre modelos (torch.cuda.empty_cache(), keep_alive=0) e índices vetoriais segregados fisicamente em disco.",
               size=12, space_before=10)

    # =========================================================================
    # SLIDE 4: Fase 1: Avaliação de Embeddings (1º Estágio)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Fase 1: Avaliação de Embeddings (1º Estágio)")

    img1_path = os.path.join(base_dir, "graficos_tcc", "ranking_hitrate_mrr_k5.png")
    if os.path.exists(img1_path):
        s4.shapes.add_picture(img1_path, Inches(5.8), Inches(1.5), width=Inches(6.8))

    tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.55), Inches(4.8), Inches(5.4))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    add_bullet(tf4, "7 Modelos Avaliados",
               "OpenAI text-embedding-3-small (FP16) vs. modelos abertos locais (Qwen3-8B/4B Q4_K_M, Multilingual-E5-Large, BGE-M3, Nomic, BERTimbau).",
               size=11, space_before=4)
    add_bullet(tf4, "Destaque Multilingual-E5-Large",
               "Alcançou a maior precisão no primeiro posto ranqueado com MRR@5 = 0.4031 e latência ultrarrápida de 22.9 ms.",
               size=11, space_before=8)
    add_bullet(tf4, "Vitória dos SLMs em Q4_K_M",
               "Qwen3-8B Q4_K_M obteve a maior cobertura do benchmark com HR@5 = 57.03% (vs. 46.88% da OpenAI; p = 0.0029).",
               size=11, space_before=8)
    add_bullet(tf4, "Retenção >100% da Acurácia",
               "Modelos locais quantizados em 4-bit compensaram a compressão com escala, entregando custo zero e 11.5x menor latência (36 ms vs. 415 ms).",
               size=11, space_before=8)

    # =========================================================================
    # SLIDE 5: O Desafio: Por que um estágio de recuperação não é suficiente?
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "O Desafio: Por que um estágio de recuperação não é suficiente?")

    tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.55), Inches(11.733), Inches(5.5))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    add_bullet(tf5, "Gargalo Estrutural dos Bi-Encoders",
               "Bi-encoders codificam pergunta e fragmentos em vetores isolados u = E(q) e v = E(d). A similaridade de cosseno condensa centenas de tokens em um único produto escalar, sendo incapaz de resolver dependências complexas (ex.: ordem de manobra de disjuntores, intertravamentos lógicos e variações entre modelos de no-breaks).",
               size=12, space_before=6)
    add_bullet(tf5, "Teto de Precisão no Top-5 do 1º Estágio",
               "Mesmo o melhor modelo de embedding (Multilingual-E5-Large) alcançou MRR@5 de 0.4031 — o que indica que em cerca de 60% das consultas o fragmento principal não é posicionado no Top-1. Chunks com alta sobreposição lexical mas irrelevantes acabam poluindo o topo.",
               size=12, space_before=12)
    add_bullet(tf5, "A Solução Arquitetural: Two-Stage RAG",
               "1º Estágio (Super-sampling K_cand = 20): Bi-encoder faz busca aproximada de alta revocação em 20–50 ms (~68% de recall).\n"
               "2º Estágio (Re-ranking Neural): Cross-Encoder ou SLM dedicado aplica atenção cruzada profunda (all-to-all cross-attention) token-a-token ou logprob softmax binário, reordenando com precisão cirúrgica no Top-5.",
               size=12, space_before=12)

    # =========================================================================
    # SLIDE 6: Fase 2: Benchmark de Re-ranking Neural no Top-5 (Tabela)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Fase 2: Benchmark de Re-ranking Neural no Top-5")

    rows, cols = 11, 8
    left, top, width, height = Inches(0.8), Inches(1.55), Inches(11.733), Inches(4.8)
    tbl_shape = s6.shapes.add_table(rows, cols, left, top, width, height)
    tbl = tbl_shape.table

    # Larguras das colunas
    col_widths = [Inches(2.3), Inches(2.2), Inches(1.0), Inches(1.1), Inches(1.1), Inches(1.0), Inches(1.6), Inches(1.4)]
    for idx, w in enumerate(col_widths):
        tbl.columns[idx].width = w

    headers_tbl = ["Modelo Base (1º)", "Reranker (2º)", "HR@5", "MRR@5", "Δ MRR@5", "Δ HR@5", "Wilcoxon (p)", "Latência"]
    for col_idx, h_text in enumerate(headers_tbl):
        cell = tbl.cell(0, col_idx)
        cell.text = h_text
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(237, 242, 247)
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(10)
            p.font.color.rgb = color_primary

    data_rerank = [
        ["multilingual_e5_large", "none (Baseline)", "54.7%", "0.4031", "—", "—", "Baseline", "0.0 ms"],
        ["multilingual_e5_large", "bge_reranker_v2_m3", "57.0%", "0.4982", "+0.0951", "+2.3%", "0.0261* (Sup.)", "1164.3 ms"],
        ["multilingual_e5_large", "qwen3_4b_q4km", "62.5%", "0.4764", "+0.0733", "+7.8%", "0.0406* (Sup.)", "6211.0 ms"],
        ["multilingual_e5_large", "qwen3_8b_q5km", "65.6% (Top)", "0.4975", "+0.0944", "+10.9%", "0.0023* (Sup.)", "6894.9 ms"],
        ["multilingual_e5_large", "rankgpt (gpt-4o-mini)", "61.7%", "0.4573", "+0.0542", "+7.0%", "0.0510 (Equiv.)", "1723.0 ms"],
        ["qwen3_8b_q4km", "none (Baseline)", "57.0%", "0.3887", "—", "—", "Baseline", "0.0 ms"],
        ["qwen3_8b_q4km", "bge_reranker_v2_m3", "53.9%", "0.4866", "+0.0979", "-3.1%", "0.0038* (Sup.)", "1145.1 ms"],
        ["qwen3_8b_q4km", "qwen3_4b_q4km", "60.9%", "0.4738", "+0.0852", "+3.9%", "0.0049* (Sup.)", "5181.6 ms"],
        ["qwen3_8b_q4km", "qwen3_8b_q5km", "60.2%", "0.5029 (Top)", "+0.1142", "+3.1%", "0.0001* (Sup.)", "6424.2 ms"],
        ["qwen3_8b_q4km", "rankgpt (gpt-4o-mini)", "57.8%", "0.4663", "+0.0776", "+0.8%", "0.0160* (Sup.)", "1647.8 ms"],
    ]

    for row_idx, row_vals in enumerate(data_rerank):
        for col_idx, val in enumerate(row_vals):
            cell = tbl.cell(row_idx + 1, col_idx)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9)
                if "(Top)" in val:
                    p.font.bold = True
                    p.font.color.rgb = color_win_green
                elif "*" in val:
                    p.font.bold = True
                    p.font.color.rgb = color_tech_blue

    # Nota explicativa
    note_box = s6.shapes.add_textbox(Inches(0.8), Inches(6.45), Inches(11.733), Inches(0.4))
    tf_n = note_box.text_frame
    p_n = tf_n.paragraphs[0]
    p_n.text = "*Nota: p < 0.05 no teste de postos com sinais de Wilcoxon comprova ganho estatisticamente significante contra o baseline puro."
    p_n.font.size = Pt(9.5)
    p_n.font.color.rgb = color_muted

    # =========================================================================
    # SLIDE 7: Descoberta Central: Modelos Locais Superam a OpenAI
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Descoberta Central: Modelos Locais Superam a OpenAI")

    img2_path = os.path.join(base_dir, "graficos_tcc", "delta_mrr5_rerankers.png")
    if os.path.exists(img2_path):
        s7.shapes.add_picture(img2_path, Inches(5.8), Inches(1.5), width=Inches(6.8))

    tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.55), Inches(4.8), Inches(5.4))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    add_bullet(tf7, "Superação da OpenAI Comercial",
               "Todos os modelos abertos locais (BGE-v2-m3, Qwen3-4B e Qwen3-8B) superaram o RankGPT (gpt-4o-mini) em ganho de MRR@5.",
               size=11, space_before=4)
    add_bullet(tf7, "Salto Expressivo de Precisão",
               "Com base Qwen3-8B, o MRR@5 saltou de 0.3887 para 0.5029 (+29.4%, delta = +0.1142) com significância estatística extrema (p = 0.0001).",
               size=11, space_before=8)
    add_bullet(tf7, "BGE-v2-m3 Superior na Base E5",
               "Alcançou MRR@5 = 0.4982 (+23.6%) contra apenas 0.4573 do RankGPT comercial na nuvem.",
               size=11, space_before=8)
    add_bullet(tf7, "Por que os Modelos Locais Vencem?",
               "Modelos treinados especificamente para relevância de passagens e com atenção cruzada possuem calibração contínua de confiança, superando prompts listwise genéricos.",
               size=11, space_before=8)

    # =========================================================================
    # SLIDE 8: Trade-off de Engenharia: Acurácia vs. Latência vs. VRAM
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Trade-off de Engenharia: Acurácia vs. Latência vs. VRAM")

    img3_path = os.path.join(base_dir, "graficos_tcc", "tradeoff_latencia_mrr_rerankers.png")
    if os.path.exists(img3_path):
        s8.shapes.add_picture(img3_path, Inches(5.8), Inches(1.5), width=Inches(6.8))

    tb8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.55), Inches(4.8), Inches(5.4))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    add_bullet(tf8, "Fronteira de Pareto Ótima",
               "BGE-Reranker-v2-m3 representa o sweet spot supremo de engenharia: adiciona apenas 1.14s a 1.16s por consulta e aloca ~1.2 GB de VRAM na GPU.",
               size=11, space_before=4)
    add_bullet(tf8, "Sweet Spot de Cobertura",
               "Qwen3-4B Q4_K_M eleva o Hit Rate@5 para 62.5% (+7.8 p.p.) com ~5.1s de latência e 2.5 GB de VRAM.",
               size=11, space_before=8)
    add_bullet(tf8, "Cobertura e Precisão Máximas",
               "Qwen3-8B Q5_K_M atinge 65.6% de HR@5 e MRR@5 de 0.5029 em ~6.4s de inferência (~5.9 GB VRAM).",
               size=11, space_before=8)
    add_bullet(tf8, "Folga de Memória Garantida",
               "Mesmo a configuração mais pesada (8B) opera estavelmente dentro dos 8 GB GDDR6 da NVIDIA RTX PRO 1000.",
               size=11, space_before=8)

    # =========================================================================
    # SLIDE 9: Recomendação de Produção: Arquitetura 100% em CPU (Sem GPU Dedicada)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Recomendação de Produção: Arquitetura 100% em CPU (Sem GPU Dedicada)")

    # Card 1: Contexto e Solução Campeã
    add_card_box(s9, Inches(0.8), Inches(1.55), Inches(5.7), Inches(4.5))
    tb_c1 = s9.shapes.add_textbox(Inches(0.95), Inches(1.65), Inches(5.4), Inches(4.3))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True
    p_c1_t = tf_c1.paragraphs[0]
    p_c1_t.text = "Contexto Operacional e Solução Campeã"
    p_c1_t.font.bold = True
    p_c1_t.font.size = Pt(13)
    p_c1_t.font.color.rgb = color_primary

    add_bullet(tf_c1, "Servidor Corporativo Exclusivo em CPU", "A infraestrutura de produção da CM Comandos Lineares opera sem placas aceleradoras GPU dedicadas.", size=11, space_before=6)
    add_bullet(tf_c1, "Solução Campeã Homologada", "1º Estágio com Multilingual-E5-Large (ONNX INT8) acelerado nativamente via instruções vetoriais AVX-512 / VNNI da CPU.", size=11, space_before=8)
    add_bullet(tf_c1, "Soberania Total e Custo Zero", "Operação 100% local (air-gapped), eliminando custos recorrentes de API e garantindo blindagem de dados confidenciais.", size=11, space_before=8)
    add_bullet(tf_c1, "Adequação Perfeita à Demanda", "Ideal para integração em sistemas internos, SAC e portais técnicos com alta estabilidade.", size=11, space_before=8)

    # Card 2: Métricas Comprovadas na CPU
    add_card_box(s9, Inches(6.833), Inches(1.55), Inches(5.7), Inches(4.5))
    tb_c2 = s9.shapes.add_textbox(Inches(6.983), Inches(1.65), Inches(5.4), Inches(4.3))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True
    p_c2_t = tf_c2.paragraphs[0]
    p_c2_t.text = "Métricas Comprovadas na CPU (Servidor CM Comandos)"
    p_c2_t.font.bold = True
    p_c2_t.font.size = Pt(13)
    p_c2_t.font.color.rgb = color_primary

    add_bullet(tf_c2, "Latência Ultrarrápida", "Apenas 107 ms por consulta na CPU pura (3,9x mais rápido que os 415 ms da API em nuvem da OpenAI!).", size=11, space_before=6)
    add_bullet(tf_c2, "Consumo Mínimo de Memória", "< 1.5 GB de RAM — viável com folga em qualquer máquina virtual ou servidor corporativo modesto.", size=11, space_before=8)
    add_bullet(tf_c2, "Precisão Superior à Nuvem", "MRR@5 de 0.4031 (já superior aos 0.3840 da OpenAI text-embedding-3-small).", size=11, space_before=8)
    add_bullet(tf_c2, "Tempo Real Assegurado", "Tempo de resposta submétrico permitindo experiência fluida para técnicos em campo.", size=11, space_before=8)

    # Rodapé do Slide 9
    add_card_box(s9, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.75))
    tb_foot9 = s9.shapes.add_textbox(Inches(0.95), Inches(6.25), Inches(11.4), Inches(0.65))
    tf_f9 = tb_foot9.text_frame
    p_f9 = tf_f9.paragraphs[0]
    p_f9.text = "Veredito de Engenharia: "
    p_f9.font.bold = True
    p_f9.font.size = Pt(11)
    p_f9.font.color.rgb = color_primary
    r_f9 = p_f9.add_run()
    r_f9.text = "O Multilingual-E5-Large (ONNX INT8) entrega 107 ms e alta precisão diretamente na CPU, tornando dispensável qualquer investimento em aceleradores gráficos para o servidor de produção."
    r_f9.font.bold = False
    r_f9.font.color.rgb = color_dark_text

    # =========================================================================
    # SLIDE 10: Estratégias Complementares e Limites de Re-ranking na CPU
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Estratégias Complementares e Limites de Re-ranking na CPU")

    # Card 1: Descarte de SLMs em CPU
    add_card_box(s10, Inches(0.8), Inches(1.55), Inches(5.7), Inches(4.5))
    tb_cpu1 = s10.shapes.add_textbox(Inches(0.95), Inches(1.65), Inches(5.4), Inches(4.3))
    tf_cpu1 = tb_cpu1.text_frame
    tf_cpu1.word_wrap = True
    p_cpu1_t = tf_cpu1.paragraphs[0]
    p_cpu1_t.text = "Limites de Engenharia: Descarte de SLMs na CPU"
    p_cpu1_t.font.bold = True
    p_cpu1_t.font.size = Pt(13)
    p_cpu1_t.font.color.rgb = color_primary

    add_bullet(tf_cpu1, "Por que SLMs Foram Descartados na CPU", "Modelos generativos de 4B e 8B (Qwen3-Reranker) exigem de 20s a 40s por consulta na CPU pura, sendo inviáveis para tempo real.", size=11, space_before=6)
    add_bullet(tf_cpu1, "Sobrecarga de Processamento", "A CPU corporativa pura sofre com o cálculo intensivo de atenção autorregressiva, gerando enfileiramento inaceitável.", size=11, space_before=8)
    add_bullet(tf_cpu1, "Foco em Eficiência Submétrica", "A arquitetura recomendada descarta SLMs no caminho crítico de consulta para assegurar tempo de resposta sob controle.", size=11, space_before=8)

    # Card 2: Estratégias Complementares
    add_card_box(s10, Inches(6.833), Inches(1.55), Inches(5.7), Inches(4.5))
    tb_cpu2 = s10.shapes.add_textbox(Inches(6.983), Inches(1.65), Inches(5.4), Inches(4.3))
    tf_cpu2 = tb_cpu2.text_frame
    tf_cpu2.word_wrap = True
    p_cpu2_t = tf_cpu2.paragraphs[0]
    p_cpu2_t.text = "Estratégias Complementares Homologadas na CPU"
    p_cpu2_t.font.bold = True
    p_cpu2_t.font.size = Pt(13)
    p_cpu2_t.font.color.rgb = color_primary

    add_bullet(tf_cpu2, "Busca Híbrida Leve (Altamente Recomendada) 🥇", "Fusão densa (E5-Large ONNX) + léxica (BM25) via Reciprocal Rank Fusion (RRF). Captura códigos de falha e peças com latência adicional de apenas +5 ms na CPU (~112 ms total).", size=11, space_before=6)
    add_bullet(tf_cpu2, "Opção com Re-ranking Neural Leve ⚡", "Cross-Encoder BGE-Reranker-v2-m3 (ONNX INT8) restrito a pool pequeno (K_cand = 5 nós). Atinge ~1.5s a 1.8s na CPU com consumo < 2.5 GB RAM.", size=11, space_before=8)
    add_bullet(tf_cpu2, "Sustentabilidade Operacional", "Ambas as soluções mantêm a premissa de zero investimento em placas aceleradoras.", size=11, space_before=8)

    # Rodapé do Slide 10
    add_card_box(s10, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.75))
    tb_foot10 = s10.shapes.add_textbox(Inches(0.95), Inches(6.25), Inches(11.4), Inches(0.65))
    tf_f10 = tb_foot10.text_frame
    p_f10 = tf_f10.paragraphs[0]
    p_f10.text = "Recomendação Final: "
    p_f10.font.bold = True
    p_f10.font.size = Pt(11)
    p_f10.font.color.rgb = color_primary
    r_f10 = p_f10.add_run()
    r_f10.text = "A Busca Híbrida (E5-Large ONNX + BM25) fornece o melhor equilíbrio na CPU da CM Comandos: máxima cobertura léxica e semântica com latência de apenas ~112 ms."
    r_f10.font.bold = False
    r_f10.font.color.rgb = color_dark_text

    # =========================================================================
    # SLIDE 11: Impacto do Projeto de ICT e Próximos Passos
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Impacto do Projeto de ICT e Próximos Passos")

    tb11 = s11.shapes.add_textbox(Inches(0.8), Inches(1.55), Inches(11.733), Inches(5.5))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    add_bullet(tf11, "Quebra de Paradigma Tecnológico",
               "Comprovação empírica e com rigor estatístico (Wilcoxon p < 0.05) de que modelos abertos locais superam soluções comerciais proprietárias em tarefas técnicas de recuperação de informação eletrotécnica.",
               size=12, space_before=6)
    add_bullet(tf11, "Viabilidade Comprovada em CPU Corporativa",
               "Demonstração prática de que o pipeline Multilingual-E5-Large em ONNX INT8 opera em apenas 107 ms na CPU pura (< 1.5 GB RAM), eliminando a necessidade de investimento em GPUs dedicadas.",
               size=12, space_before=10)
    add_bullet(tf11, "Soberania de Dados e Custo Zero Recorrente",
               "Eliminação total de custos recorrentes de APIs comerciais e blindagem estrita de segredos industriais e manuais confidenciais de no-breaks.",
               size=12, space_before=10)
    add_bullet(tf11, "Próxima Etapa: Otimização e Validação em CPU",
               "Testes de concorrência e carga do pipeline E5-Large ONNX + BM25 na infraestrutura de servidores da CM Comandos Lineares.",
               size=12, space_before=10)
    add_bullet(tf11, "Piloto Operacional Integrado",
               "Desenvolvimento de protótipo funcional para testes práticos pela equipe de assistência técnica e engenharia de suporte da CM Comandos Lineares.",
               size=12, space_before=10)

    # =========================================================================
    # SLIDE 12: Perguntas & Discussão Técnica
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    box12 = s12.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.5))
    tf12 = box12.text_frame
    tf12.word_wrap = True

    p12 = tf12.paragraphs[0]
    p12.text = "Perguntas & Discussão Técnica"
    p12.font.size = Pt(32)
    p12.font.bold = True
    p12.font.color.rgb = color_primary

    p12_sub = tf12.add_paragraph()
    p12_sub.text = "Avaliação Científica e Tecnológica de RAG Industrial (Retrieval & Re-ranking)"
    p12_sub.font.size = Pt(18)
    p12_sub.font.bold = True
    p12_sub.font.color.rgb = color_tech_blue
    p12_sub.space_before = Pt(14)

    p12_det = tf12.add_paragraph()
    p12_det.text = "Autor: João Messias Lima Pereira  |  Projeto de Pesquisa Científica e Tecnológica (ICT)\nHardware de Referência: Intel Core Ultra 7 (CPU) & NVIDIA RTX PRO 1000 (8 GB GDDR6)  |  CM Comandos Lineares\n\nRepositório de dados, pipelines e relatórios técnicos disponíveis para replicação aberta."
    p12_det.font.size = Pt(12)
    p12_det.font.color.rgb = color_muted
    p12_det.space_before = Pt(24)

    prs.save(output_path)


def main() -> None:
    base_dir = os.path.abspath(".")
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    # 1. Gerar Markdown (Marp)
    md_path = os.path.join(docs_dir, "apresentacao_pesquisa_reuniao.md")
    marp_text = build_marp_markdown(base_dir)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(marp_text)
    print(f"[OK] Marp Markdown gerado: {md_path} ({len(marp_text)} chars)")

    # 2. Compilar HTML via Marp CLI
    html_path = os.path.join(docs_dir, "apresentacao_pesquisa_reuniao.html")
    cmd_html = ["npx", "@marp-team/marp-cli", "--html", "--allow-local-files", md_path, "-o", html_path]
    r_html = subprocess.run(cmd_html, capture_output=True, text=True)
    if r_html.returncode != 0:
        print(f"[ERRO] Falha ao compilar Marp HTML: {r_html.stderr}")
    else:
        print(f"[OK] Marp HTML gerado: {html_path} ({os.path.getsize(html_path):,} bytes)")

    # 3. Compilar PDF via Marp CLI
    pdf_path = os.path.join(docs_dir, "apresentacao_pesquisa_reuniao.pdf")
    cmd_pdf = ["npx", "@marp-team/marp-cli", "--html", "--allow-local-files", md_path, "--pdf", "-o", pdf_path]
    r_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True)
    if r_pdf.returncode != 0:
        print(f"[ERRO] Falha ao compilar Marp PDF: {r_pdf.stderr}")
    else:
        print(f"[OK] Marp PDF gerado: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")

    # 4. Gerar PowerPoint editável (.pptx) via python-pptx
    pptx_path = os.path.join(docs_dir, "apresentacao_pesquisa_reuniao.pptx")
    build_pptx_presentation(pptx_path, base_dir)
    print(f"[OK] PowerPoint gerado: {pptx_path} ({os.path.getsize(pptx_path):,} bytes)")

    # 5. Copiar PDF para a raiz do repositório
    root_pdf_path = os.path.join(base_dir, "apresentacao_pesquisa_reuniao.pdf")
    shutil.copy2(pdf_path, root_pdf_path)
    print(f"[OK] Cópia para a raiz: {root_pdf_path} ({os.path.getsize(root_pdf_path):,} bytes)")


if __name__ == "__main__":
    main()
