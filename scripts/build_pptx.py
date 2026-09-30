import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

base_dir = os.path.abspath(".")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

def add_header(slide, title_text, category="AVALIAÇÃO DE EMBEDDINGS RAG INDUSTRIAL"):
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category.upper()
    p_c.font.size = Pt(11)
    p_c.font.bold = True
    p_c.font.color.rgb = RGBColor(43, 108, 176)
    
    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf = t_box.text_frame
    tf.word_wrap = True
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
    p.space_before = Pt(10)
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.color.rgb = RGBColor(45, 55, 72)

# Slide 1: Cover
s1 = prs.slides.add_slide(blank_layout)
tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.8))
tf = tb.text_frame
p1 = tf.paragraphs[0]
p1.text = "Avaliação Comparativa de Modelos de Embedding"
p1.font.size = Pt(32)
p1.font.bold = True
p1.font.color.rgb = RGBColor(26, 54, 93)

p2 = tf.add_paragraph()
p2.text = "Recuperação de Informação Técnica de Nobreaks: Baseline OpenAI vs. Modelos Locais Quantizados em 4-bit (Q4_K_M)"
p2.font.size = Pt(18)
p2.font.color.rgb = RGBColor(43, 108, 176)
p2.space_before = Pt(14)

p3 = tf.add_paragraph()
p3.text = "Autor: João Vitor Rura  |  Projeto de Pesquisa Científica e Tecnológica (ICT)\nHardware: NVIDIA RTX PRO 1000 Laptop GPU (8 GB GDDR6)  |  Setembro / 2026"
p3.font.size = Pt(13)
p3.font.color.rgb = RGBColor(113, 128, 150)
p3.space_before = Pt(28)

# Slide 2: Objectives
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "Objetivos e Formalização da Hipótese H1")
tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.0))
tf2 = tb2.text_frame
tf2.word_wrap = True
add_bullet(tf2, "Objetivo Central", "Validar a substituição de modelos de embedding comerciais proprietários (OpenAI text-embedding-3-small) por modelos abertos locais operando em quantização 4-bit (Q4_K_M) em GPU de 8 GB.")
add_bullet(tf2, "Hipótese Alternativa (H1)", "Pelo menos um modelo aberto local quantizado em Q4_K_M retém ≥ 85% do desempenho do baseline comercial em ≥ 2 métricas de IR no ponto de corte Top-5 (Hit Rate@5, MRR@5, Recall@5, MAP@5).")
add_bullet(tf2, "Critérios Estatísticos", "Aplicação do teste não-paramétrico pareado de Wilcoxon (alpha=0.05) para avaliação de equivalência ou superioridade, complementado por intervalos de confiança Bootstrap de 95% (B=1.000).")
add_bullet(tf2, "Garantias de Engenharia", "Custo zero de chamadas de API, latência inferior a 100 ms por consulta e soberania absoluta de dados industriais sensíveis da CM Comandos.")

# Slide 3: Results Table
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "Resultados Consolidados no Top-5 (K=5)")
rows, cols = 8, 8
left, top, width, height = Inches(0.8), Inches(1.6), Inches(11.7), Inches(4.8)
table_shape = s3.shapes.add_table(rows, cols, left, top, width, height)
table = table_shape.table

headers = ["Modelo", "Quant.", "Dim.", "HR@5", "MRR@5", "Recall@5", "Latência", "Wilcoxon p (MRR)"]
for col_idx, h in enumerate(headers):
    cell = table.cell(0, col_idx)
    cell.text = h
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(237, 242, 247)
    for p in cell.text_frame.paragraphs:
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(26, 54, 93)

data_table = [
    ["multilingual_e5_large", "FP16", "1024", "0.5469", "0.4031 (1º)", "0.5110", "22.88 ms", "p = 0.5217 (Equiv.)"],
    ["qwen3_8b_q4km", "Q4_K_M", "4096", "0.5703 (1º)", "0.3887 (2º)", "0.5311 (1º)", "57.06 ms", "p = 0.9336 (Equiv.)"],
    ["qwen3_4b_q4km", "Q4_K_M", "2560", "0.5547", "0.3844", "0.4965", "35.98 ms", "p = 0.9565 (Equiv.)"],
    ["openai_3_small (Base)", "FP16", "1536", "0.4688", "0.3840", "0.4482", "415.07 ms", "— (Baseline)"],
    ["bge_m3", "FP16", "1024", "0.5234", "0.3797", "0.4673", "167.30 ms", "p = 0.8273 (Equiv.)"],
    ["nomic_embed_text", "Q4_K_M", "768", "0.3594", "0.3021", "0.3466", "9.78 ms", "p = 0.0142* (Inferior)"],
    ["bertimbau_sts", "FP16", "768", "0.4531", "0.2423", "0.3498", "35.67 ms", "p = 0.0008* (Inferior)"]
]

for row_idx, row_data in enumerate(data_table):
    for col_idx, val in enumerate(row_data):
        cell = table.cell(row_idx + 1, col_idx)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            if "Q4_K_M" in val:
                p.font.bold = True
                p.font.color.rgb = RGBColor(43, 108, 176)
            elif "(1º)" in val:
                p.font.bold = True
                p.font.color.rgb = RGBColor(34, 139, 34)

# Slide 4: Chart 1
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "Eficácia de Recuperação: MRR@5 vs. Hit Rate@5")
s4.shapes.add_picture(f"{base_dir}/graficos_tcc/ranking_hitrate_mrr_k5.png", Inches(0.8), Inches(1.5), width=Inches(7.5))
tb4 = s4.shapes.add_textbox(Inches(8.6), Inches(1.8), Inches(4.0), Inches(4.8))
tf4 = tb4.text_frame
tf4.word_wrap = True
add_bullet(tf4, "Liderança de MRR", "multilingual-e5-large alcançou o maior MRR@5 (0.4031), seguido pelo Qwen3-8B Q4_K_M (0.3887).")
add_bullet(tf4, "Liderança de Hit Rate", "Qwen3-8B Q4_K_M obteve a maior cobertura de nós relevantes (57.03%), superando a OpenAI (46.88%).")
add_bullet(tf4, "Validação H1", "Ambos os modelos Qwen3 Q4_K_M superaram o baseline OpenAI em HR@5 e MRR@5.")

# Slide 5: Chart 2
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "Curva de Evolução do Recall por Nível de Top-K")
s5.shapes.add_picture(f"{base_dir}/graficos_tcc/curva_recuperacao_topk.png", Inches(0.8), Inches(1.5), width=Inches(7.5))
tb5 = s5.shapes.add_textbox(Inches(8.6), Inches(1.8), Inches(4.0), Inches(4.8))
tf5 = tb5.text_frame
tf5.word_wrap = True
add_bullet(tf5, "Padrão de Cobertura", "A evolução K=2 -> K=5 -> K=10 mostra separação clara em 3 clusters distintos de capacidade.")
add_bullet(tf5, "Cluster de Alta Eficácia", "E5-Large, Qwen3-8B, Qwen3-4B e BGE-M3 formam o grupo de elite com Recall@10 atingindo 56% a 60%.")
add_bullet(tf5, "OpenAI no Ponto Médio", "O modelo da OpenAI estabiliza em 48% de Recall@10, sendo ultrapassado pelos modelos abertos locais.")

# Slide 6: Chart 3
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "Trade-off de Engenharia: Eficiência (ms) vs. Eficácia (MRR@5)")
s6.shapes.add_picture(f"{base_dir}/graficos_tcc/tradeoff_latencia_mrr.png", Inches(0.8), Inches(1.5), width=Inches(7.5))
tb6 = s6.shapes.add_textbox(Inches(8.6), Inches(1.8), Inches(4.0), Inches(4.8))
tf6 = tb6.text_frame
tf6.word_wrap = True
add_bullet(tf6, "Quadrante Ótimo", "Modelos no canto superior esquerdo combinam alta eficácia semântica e latência extremamente reduzida.")
add_bullet(tf6, "Ganho de Velocidade", "Qwen3-4B Q4_K_M é 11.5x mais rápido que a OpenAI (36 ms vs. 415 ms) devido à ausência de overhead de rede.")
add_bullet(tf6, "Nomic Ultraleve", "Alcança 9.8 ms/query, mas sofre perda acentuada de MRR (0.3021 vs. 0.3840).")

# Slide 7: Conclusions
s7 = prs.slides.add_slide(blank_layout)
add_header(s7, "Conclusões e Recomendações de Engenharia")
tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.0))
tf7 = tb7.text_frame
tf7.word_wrap = True
add_bullet(tf7, "Confirmação Plena da Hipótese H1", "Os modelos locais quantizados em 4 bits (Q4_K_M) não apenas atingiram 85% do baseline OpenAI, como o superaram em Hit Rate, Recall e MRR com equivalência estatística comprovada (Wilcoxon p > 0.90).")
add_bullet(tf7, "Sweet Spot da Quantização", "A escala de parâmetros (7.6B e 4.0B) compensa plenamente a quantização em 4 bits, mantendo a geometria esférica do espaço vetorial para termos técnicos de engenharia de nobreaks.")
add_bullet(tf7, "Recomendação para Produção na CM Comandos", "Adotar o Qwen3-4B Q4_K_M (para deploy unificado no Ollama com 2.9 GB VRAM e 36 ms de latência) ou o Multilingual-E5-Large (para maximização de MRR com 22.9 ms).")
add_bullet(tf7, "Impacto Corporativo", "Custo zero recorrente de API, latência até 11.5x menor e garantia absoluta de confidencialidade industrial sobre esquemáticos e manuais proprietários.")

pptx_path = "docs/apresentacao_embeddings.pptx"
prs.save(pptx_path)
print(f"PowerPoint Presentation saved successfully: {pptx_path}")
