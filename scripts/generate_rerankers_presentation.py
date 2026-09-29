"""Gera a apresentação completa de Re-ranking Neural (Two-Stage RAG) para o TCC.

Produz:
1. 'docs/apresentacao_rerankers.md' (Marp Markdown com layout moderno e 16:9)
2. 'docs/apresentacao_rerankers.html' (Compilado via @marp-team/marp-cli)
3. 'docs/apresentacao_rerankers.pdf' (Compilado via @marp-team/marp-cli --pdf)
4. 'docs/apresentacao_rerankers.pptx' (Apresentação PowerPoint editável via python-pptx)
"""

from __future__ import annotations

import os
import subprocess
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor


def generate_rerankers_presentation() -> None:
    base_dir = os.path.abspath(".")
    csv_path = os.path.join(base_dir, "results", "consolidated_reranker_metrics.csv")
    
    # Gera a tabela dinâmica a partir de consolidated_reranker_metrics.csv
    table_lines = [
        "| Modelo Base | Reranker | HR@5 | MRR@5 | Δ MRR@5 | Δ HR@5 | Wilcoxon (p) | Latência |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            base = row["base_model"]
            rerank = row["reranker"]
            hr5 = f"{row.get('hit_rate@5_mean', 0.0) * 100:.1f}%"
            mrr5 = f"{row.get('mrr@5_mean', 0.0):.4f}"
            d_mrr5 = f"{row.get('delta_mrr@5', 0.0):+.4f}" if rerank != "none" else "—"
            d_hr5 = f"{row.get('delta_hit_rate@5', 0.0) * 100:+.1f}%" if rerank != "none" else "—"
            p_val = f"{row.get('p_value_wilcoxon_mrr5', 1.0):.4f}" if rerank != "none" else "Baseline"
            lat = f"{row.get('mean_latency_ms', 0.0):.1f} ms"
            table_lines.append(f"| `{base}` | `{rerank}` | {hr5} | **{mrr5}** | {d_mrr5} | {d_hr5} | {p_val} | {lat} |")
    marp_table = "\n".join(table_lines)

    # 1. Monta o conteúdo Marp Markdown
    marp_content = f"""---
marp: true
theme: default
paginate: true
header: "Re-ranking Neural em RAG Industrial (Two-Stage RAG) — TCC"
footer: "CM Comandos Lineares | NVIDIA RTX PRO 1000 (8 GB GDDR6)"
style: |
  section {{
    font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 30px 45px;
    background-color: #ffffff;
    color: #1a202c;
  }}
  h1 {{
    color: #1a365d;
    font-size: 1.6em;
    margin-bottom: 0.2em;
    margin-top: 10px;
  }}
  h2 {{
    color: #2b6cb0;
    font-size: 1.2em;
    margin-top: 15px;
    margin-bottom: 0.3em;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
  }}
  p, li {{
    font-size: 0.72em;
    line-height: 1.45;
  }}
  ul {{
    margin-top: 6px;
    margin-bottom: 8px;
  }}
  table {{
    font-size: 0.60em;
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
  }}
  th {{
    background-color: #edf2f7;
    color: #2d3748;
    padding: 5px 6px;
  }}
  td {{
    padding: 4px 6px;
    border-bottom: 1px solid #e2e8f0;
  }}
  .badge-win {{
    background-color: #c6f6d5;
    color: #22543d;
    font-weight: bold;
    padding: 1px 5px;
    border-radius: 3px;
  }}
  .badge-highlight {{
    background-color: #bee3f8;
    color: #2b6cb0;
    font-weight: bold;
    padding: 1px 5px;
    border-radius: 3px;
  }}
  .grid-2 {{
    display: flex;
    gap: 20px;
    align-items: center;
    margin-top: 10px;
  }}
  .col-text {{
    flex: 1;
  }}
  .col-img {{
    flex: 1.2;
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
  footer {{
    font-size: 0.5em;
    color: #718096;
  }}
  header {{
    font-size: 0.52em;
    color: #718096;
  }}
---

<!-- _class: lead -->
# Avaliação de Re-ranking Neural em Dois Estágios (Two-Stage RAG)
### Otimização da Precisão em Manuais Técnicos Industriais com Modelos Quantizados Locais

**Trabalho de Conclusão de Curso (TCC) — Engenharia de Software / Ciência da Computação**
*Hardware de Execução:* NVIDIA RTX PRO 1000 Blackwell (8 GB GDDR6) | Intel Core Ultra 7 265H | 32 GB RAM
*Estudo de Caso:* 16 Manuais de No-breaks Industriais — CM Comandos Lineares

---

## 1. Motivação: Limitações do RAG em Estágio Único

- **O Desafio do Jargão Industrial**:
  - Manuais técnicos contêm termos como *bypass estático*, *retificador tiristorizado*, *inversor IGBT*, *tensão de flutuação VRLA*.
  - Bi-encoders de 1º estágio operam em representação vetorial independente, perdendo nuances contextuais finas entre a pergunta e o fragmento.
- **A Solução Two-Stage RAG**:
  - **1º Estágio (Super-sampling $K_{{\\text{{cand}}}}=20$)**: Recupera rapidamente uma vizinhança ampla de alta revocação (*recall*).
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
  - Teste pareado não-paramétrico de **Wilcoxon** ($\alpha = 0.05$) sobre a variação consulta-a-consulta de MRR@5.
  - Intervalos de Confiança **Bootstrap 95%** ($B = 1.000$ iterações).
- **Gestão Estrita de VRAM**:
  - Descarga obrigatória de memória GPU entre modelos (`torch.cuda.empty_cache()`, `keep_alive=0`, `gc.collect()`).

---

## 4. Tabela Geral de Resultados no Top-5 ($K=5$)

{marp_table}

*\*Nota: p < 0.05 comprova significância estatística do ganho contra o baseline puro correspondente.*


---

## 5. Eficácia de Recuperação: MRR@5 vs. Hit Rate@5

<div class="grid-2">
<div class="col-text">

- **Salto Expressivo de Precisão**:
  - O re-ranking elevou o MRR@5 de **0.3887 $\\rightarrow$ 0.5029 (+29.4%)** na base Qwen3-8B.
  - Na base E5-Large, o MRR@5 saltou de **0.4031 $\\rightarrow$ 0.4980 (+23.5%)**.
- **Resgate de Documentos Relevantes**:
  - O Hit Rate@5 aumentou até **+7.8 pontos percentuais** (de 54.7% para 62.5% com Qwen3-4B).
- **Top Performer Absoluto**:
  - `Qwen3-8B (Q5_K_M)` entregou a melhor ordenação média do experimento (MRR@5 = 0.5029).

</div>
<div class="col-img">
<img src="{base_dir}/graficos_tcc/ranking_rerankers_mrr_hr_k5.png" class="chart-img" alt="Ranking MRR5 e HR5">
</div>
</div>

---

## 6. Ganhos Incrementais: Todos os Modelos Locais Superam RankGPT

<div class="grid-2">
<div class="col-text">

- **Superação da OpenAI Comercial**:
  - `RankGPT (gpt-4o-mini)` obteve $\\Delta$ MRR@5 de **+0.0776**.
  - `Qwen3-4B (Q4_K_M)` obteve $\\Delta$ MRR@5 de **+0.0852**.
  - `BGE-v2-m3` obteve $\\Delta$ MRR@5 de **+0.0979**.
  - `Qwen3-8B (Q5_K_M)` obteve $\\Delta$ MRR@5 de **+0.1142**.
- **Por que os Modelos Locais Vencem?**:
  - *Fine-tuning dedicado para relevância de passagens* supera o prompt genérico listwise da LLM.
  - Logprobs em token binário ("yes"/"no") capturam calibração fina de confiança semântica.

</div>
<div class="col-img">
<img src="{base_dir}/graficos_tcc/delta_mrr5_rerankers.png" class="chart-img" alt="Ganhos Incrementais MRR5">
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
<img src="{base_dir}/graficos_tcc/tradeoff_latencia_mrr_rerankers.png" class="chart-img" alt="Trade-off Latencia vs MRR5">
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
   - **1º Estágio**: `Multilingual-E5-Large` (FP16 ou ONNX INT8) $\rightarrow$ 22.9 ms.
   - **2º Estágio**: `BGE-Reranker-v2-m3` $\rightarrow$ 1145 ms.
   - **Tempo Total**: ~1.17 segundos | **MRR@5**: 0.4866 | **VRAM Máxima**: 2.5 GB.

2. **Cenário de Análise Aprofundada / Suporte Especializado**:
   - **1º Estágio**: `Qwen3-Embedding-8B (Q4_K_M)` $\rightarrow$ 57.1 ms.
   - **2º Estágio**: `Qwen3-Reranker-8B (Q5_K_M)` $\rightarrow$ 6424 ms.
   - **Tempo Total**: ~6.48 segundos | **MRR@5**: **0.5029** | **HR@5**: **60.2%**.

---

<!-- _class: lead -->
# Perguntas & Discussão Técnica

### Contribuições do TCC:
- Demonstração empírica da viabilidade de RAG Two-Stage 100% local em GPU de 8 GB.
- Confirmação de que SLMs quantizados locais superam modelos comerciais genéricos em domínio eletrotécnico.
- Código, dados e pipelines totalmente reprodutíveis.

*Obrigado!*
"""

    marp_path = os.path.join(base_dir, "docs", "apresentacao_rerankers.md")
    with open(marp_path, "w", encoding="utf-8") as f:
        f.write(marp_content)
    print(f"Marp Markdown gerado em '{marp_path}'.")

    # 2. Compilação via Marp CLI
    html_out = os.path.join(base_dir, "docs", "apresentacao_rerankers.html")
    pdf_out = os.path.join(base_dir, "docs", "apresentacao_rerankers.pdf")

    cmd_html = ["npx", "@marp-team/marp-cli", "--html", marp_path, "-o", html_out]
    r_html = subprocess.run(cmd_html, capture_output=True, text=True)
    print(f"Marp HTML status: {r_html.returncode} -> {html_out}")

    cmd_pdf = ["npx", "@marp-team/marp-cli", "--html", marp_path, "--pdf", "-o", pdf_out]
    r_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True)
    print(f"Marp PDF status: {r_pdf.returncode} -> {pdf_out}")

    # 3. Geração do PowerPoint (.pptx) editável via python-pptx
    pptx_out = os.path.join(base_dir, "docs", "apresentacao_rerankers.pptx")
    build_rerankers_pptx(pptx_out, base_dir)
    print(f"PowerPoint gerado em '{pptx_out}'.")


def build_rerankers_pptx(output_path: str, base_dir: str) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category="AVALIAÇÃO DE RE-RANKING NEURAL (TWO-STAGE RAG) — TCC"):
        c_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
        tf_c = c_box.text_frame
        p_c = tf_c.paragraphs[0]
        p_c.text = category.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = RGBColor(43, 108, 176)

        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        tf = t_box.text_frame
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = RGBColor(26, 54, 93)

    def add_bullet(tf, title, body):
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = title + ": "
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = RGBColor(43, 108, 176)
        p.space_before = Pt(8)
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.color.rgb = RGBColor(26, 32, 44)

    # Slide 1: Capa
    s1 = prs.slides.add_slide(blank_layout)
    box1 = s1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.0))
    tf1 = box1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Avaliação de Re-ranking Neural em Dois Estágios (Two-Stage RAG)"
    p1.font.size = Pt(30)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(26, 54, 93)

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Otimização da Precisão em Manuais Industriais com Modelos Locais Quantizados"
    p1_sub.font.size = Pt(18)
    p1_sub.font.color.rgb = RGBColor(43, 108, 176)
    p1_sub.space_before = Pt(12)

    p1_meta = tf1.add_paragraph()
    p1_meta.text = "Hardware: NVIDIA RTX PRO 1000 (8 GB GDDR6) | Estudo de Caso: 16 Manuais CM Comandos Lineares"
    p1_meta.font.size = Pt(13)
    p1_meta.font.color.rgb = RGBColor(113, 128, 150)
    p1_meta.space_before = Pt(24)

    # Slide 2: Motivação
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Motivação e Arquitetura Two-Stage RAG")
    box2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf2 = box2.text_frame
    tf2.word_wrap = True
    add_bullet(tf2, "Desafio Técnico", "Jargão eletrotécnico denso (bypass estático, retificador tiristorizado, VRLA) exige atenção fina entre tokens.")
    add_bullet(tf2, "Arquitetura Two-Stage", "O 1º estágio recupera 20 candidatos (alta revocação); o 2º estágio reordena com atenção cruzada profunda.")
    add_bullet(tf2, "Objetivo Central", "Demonstrar que modelos locais quantizados (Q4_K_M / Q5_K_M) superam APIs comerciais com custo financeiro zero.")

    # Slide 3: Modelos
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Matriz Experimental de Modelos")
    box3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf3 = box3.text_frame
    tf3.word_wrap = True
    add_bullet(tf3, "1º Estágio (Dense Embeddings)", "Top-2 do benchmark prévio: Qwen3-Embedding-8B (Q4_K_M) e Multilingual-E5-Large (FP16).")
    add_bullet(tf3, "2º Estágio - SLMs Quantizados", "Qwen3-Reranker-4B (Q4_K_M) e Qwen3-Reranker-8B (Q5_K_M) via Ollama com logprob softmax binário.")
    add_bullet(tf3, "2º Estágio - Cross-Encoder Local", "BAAI/bge-reranker-v2-m3 (HuggingFace PyTorch, ~1.2 GB VRAM).")
    add_bullet(tf3, "2º Estágio - Baseline Comercial", "RankGPT baseado em OpenAI gpt-4o-mini (Prompt listwise via API).")

    # Slide 4: Gráfico 1 (Ranking MRR5 vs HR5)
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Impacto no Top-5: MRR@5 vs. Hit Rate@5")
    img1_path = os.path.join(base_dir, "graficos_tcc", "ranking_rerankers_mrr_hr_k5.png")
    if os.path.exists(img1_path):
        s4.shapes.add_picture(img1_path, Inches(4.8), Inches(1.8), width=Inches(7.8))
    box4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(3.8), Inches(5.0))
    tf4 = box4.text_frame
    tf4.word_wrap = True
    add_bullet(tf4, "Salto em MRR@5", "Elevação de 0.3887 para 0.5029 (+29.4%) na base Qwen3-8B.")
    add_bullet(tf4, "Resgate de Relevantes", "Hit Rate@5 atingiu 62.5% (+7.8 p.p.) com Qwen3-4B.")
    add_bullet(tf4, "Top Performer", "Qwen3-8B Q5_K_M entregou o maior MRR@5 absoluto (0.5029).")

    # Slide 5: Gráfico 2 (Delta MRR5)
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Ganhos Incrementais: Modelos Locais Superam RankGPT")
    img2_path = os.path.join(base_dir, "graficos_tcc", "delta_mrr5_rerankers.png")
    if os.path.exists(img2_path):
        s5.shapes.add_picture(img2_path, Inches(4.8), Inches(1.8), width=Inches(7.8))
    box5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(3.8), Inches(5.0))
    tf5 = box5.text_frame
    tf5.word_wrap = True
    add_bullet(tf5, "OpenAI Superada", "RankGPT obteve +0.0776, enquanto Qwen3-4B (+0.0852), BGE (+0.0979) e Qwen3-8B (+0.1142) foram superiores.")
    add_bullet(tf5, "Por que vencem?", "Modelos dedicados a passage-ranking superam a ordenação por prompt genérico.")

    # Slide 6: Gráfico 3 (Trade-off Latência vs MRR)
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Trade-off de Engenharia: Acurácia vs. Latência")
    img3_path = os.path.join(base_dir, "graficos_tcc", "tradeoff_latencia_mrr_rerankers.png")
    if os.path.exists(img3_path):
        s6.shapes.add_picture(img3_path, Inches(4.8), Inches(1.8), width=Inches(7.8))
    box6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(3.8), Inches(5.0))
    tf6 = box6.text_frame
    tf6.word_wrap = True
    add_bullet(tf6, "Sweet Spot Latência", "BGE-Reranker-v2-m3 adiciona apenas 1.14s por consulta na GPU.")
    add_bullet(tf6, "Sweet Spot Cobertura", "Qwen3-4B Q4_K_M atinge 62.5% de HR@5 com 5.1s de latência.")
    add_bullet(tf6, "SLA Industrial", "BGE-v2-m3 viabiliza tempos de resposta abaixo de 1.5s ponta a ponta.")

    # Slide 7: Conclusões e Recomendação
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Conclusões e Recomendação para a CM Comandos")
    box7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf7 = box7.text_frame
    tf7.word_wrap = True
    add_bullet(tf7, "Recomendação Produção", "Pipeline Híbrido: Multilingual-E5-Large (1º estágio) + BGE-Reranker-v2-m3 (2º estágio).")
    add_bullet(tf7, "Vantagens Chave", "Latência total ~1.17s, VRAM total < 2.5 GB, custo zero de API, soberania total dos dados.")
    add_bullet(tf7, "Cenário Máxima Precisão", "Qwen3-8B Q4_K_M + Qwen3-8B Q5_K_M atinge teto de MRR@5 = 0.5029.")

    prs.save(output_path)


if __name__ == "__main__":
    generate_rerankers_presentation()
