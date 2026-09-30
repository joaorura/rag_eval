import os
import markdown
import re
import subprocess

with open("docs/analise_comparativa_resultados.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# Make image paths absolute for WeasyPrint
base_dir = os.path.abspath(".")
md_text = md_text.replace("../graficos_tcc/", f"{base_dir}/graficos_tcc/")
md_text = md_text.replace("graficos_tcc/", f"{base_dir}/graficos_tcc/")

html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'codehilite'])

html_full = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Análise Comparativa de Modelos de Embedding</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 15mm 20mm 15mm;
        @bottom-right {{
            content: "Página " counter(page) " de " counter(pages);
            font-size: 8pt;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #666;
        }}
        @bottom-left {{
            content: "Pesquisa Científica e Tecnológica Aplicada (ICT) - CM Comandos RAG Eval";
            font-size: 8pt;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #666;
        }}
    }}
    body {{
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 9.5pt;
        line-height: 1.5;
        color: #222;
    }}
    h1 {{
        color: #1a365d;
        font-size: 18pt;
        border-bottom: 2px solid #2b6cb0;
        padding-bottom: 4px;
        margin-top: 0;
        margin-bottom: 12px;
    }}
    h2 {{
        color: #2b6cb0;
        font-size: 13pt;
        border-bottom: 1px solid #cbd5e0;
        padding-bottom: 3px;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }}
    h3 {{
        color: #2d3748;
        font-size: 11pt;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}
    h4 {{
        color: #4a5568;
        font-size: 10pt;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 8px;
        text-align: justify;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0 14px 0;
        font-size: 8pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #e2e8f0;
        padding: 5px 6px;
        text-align: left;
    }}
    th {{
        background-color: #edf2f7;
        color: #2d3748;
        font-weight: bold;
    }}
    tr:nth-child(even) {{
        background-color: #f7fafc;
    }}
    blockquote {{
        border-left: 3px solid #3182ce;
        padding: 4px 10px;
        margin: 8px 0;
        background-color: #ebf8ff;
        color: #2c5282;
        font-size: 9pt;
    }}
    code {{
        font-family: 'Courier New', Courier, monospace;
        background-color: #edf2f7;
        padding: 1px 3px;
        border-radius: 3px;
        font-size: 8.5pt;
    }}
    pre {{
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 8px;
        font-size: 8pt;
        overflow-x: auto;
        page-break-inside: avoid;
    }}
    img {{
        max-width: 95%;
        height: auto;
        display: block;
        margin: 10px auto;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        page-break-inside: avoid;
    }}
    hr {{
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 16px 0;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

html_path = "docs/relatorio_comparativo_embeddings.html"
pdf_path = "docs/relatorio_comparativo_embeddings.pdf"

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_full)

print(f"HTML saved: {html_path}")

cmd = ["/home/joaorura/.local/bin/weasyprint", html_path, pdf_path]
res = subprocess.run(cmd, capture_output=True, text=True)
print("WeasyPrint return code:", res.returncode)
if res.stderr:
    print("Stderr snippet:", res.stderr[:400])
if os.path.exists(pdf_path):
    print(f"PDF generated successfully: {pdf_path} (Size: {os.path.getsize(pdf_path)/1024:.1f} KB)")
else:
    print("PDF generation failed!")
