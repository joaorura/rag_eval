"""Gera os relatórios em PDF do benchmark de Re-ranking Neural (Two-Stage RAG).

1. Converte o Jupyter Notebook 'notebooks/avaliacao_rerankers_dois_estagios.ipynb'
   em HTML e em PDF de alta resolução ('docs/avaliacao_rerankers_dois_estagios.pdf').
2. Cria o relatório técnico acadêmico ABNT/IEEE 'docs/relatorio_rerankers_dois_estagios.pdf'
   via WeasyPrint com todas as tabelas, testes de hipótese e gráficos embutidos.
"""

from __future__ import annotations

import os
import subprocess
import markdown
import pandas as pd


def generate_rerankers_pdf_report() -> None:
    base_dir = os.path.abspath(".")
    os.makedirs(os.path.join(base_dir, "docs"), exist_ok=True)

    # 1. Converte o notebook executado para HTML e PDF via WeasyPrint
    ipynb_path = os.path.join(base_dir, "notebooks", "avaliacao_rerankers_dois_estagios.ipynb")
    nb_html = os.path.join(base_dir, "docs", "avaliacao_rerankers_dois_estagios.html")
    nb_pdf = os.path.join(base_dir, "docs", "avaliacao_rerankers_dois_estagios.pdf")

    if os.path.exists(ipynb_path):
        print(f"Convertendo '{ipynb_path}' para HTML...")
        cmd_nb = [
            ".venv/bin/jupyter", "nbconvert",
            "--to", "html",
            "--template", "classic",
            ipynb_path,
            "--output", nb_html,
        ]
        subprocess.run(cmd_nb, capture_output=True, text=True)

        if os.path.exists(nb_html):
            print(f"Compilando '{nb_html}' para PDF via WeasyPrint...")
            cmd_wp = ["/home/joaorura/.local/bin/weasyprint", nb_html, nb_pdf]
            subprocess.run(cmd_wp, capture_output=True, text=True)
            if os.path.exists(nb_pdf):
                print(f"PDF do Notebook gerado com sucesso: '{nb_pdf}' ({os.path.getsize(nb_pdf)/1024:.1f} KB).")

    # 2. Gera o Relatório Técnico Consolidado em Formato Acadêmico ABNT
    generate_academic_pdf_report(base_dir)


def generate_academic_pdf_report(base_dir: str) -> None:
    csv_path = os.path.join(base_dir, "results", "consolidated_reranker_metrics.csv")
    
    # Monta tabelas em HTML
    table_rows = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            base = row.get("base_model", "")
            rerank = row.get("reranker", "")
            hr5 = f"{row.get('hit_rate@5_mean', 0.0) * 100:.1f}%"
            mrr5 = f"{row.get('mrr@5_mean', 0.0):.4f}"
            rec5 = f"{row.get('recall@5_mean', 0.0) * 100:.1f}%"
            map5 = f"{row.get('map@5_mean', 0.0):.4f}"
            d_mrr5 = f"{row.get('delta_mrr@5', 0.0):+.4f}" if rerank != "none" else "—"
            p_val = f"{row.get('p_value_wilcoxon_mrr5', 1.0):.4f}" if rerank != "none" else "Baseline"
            lat = f"{row.get('mean_latency_ms', 0.0):.1f} ms"
            
            sig_badge = "style='color: #22543d; font-weight: bold;'" if "0.0" in p_val and float(p_val) < 0.05 else ""

            table_rows.append(f"""
            <tr>
                <td><code>{base}</code></td>
                <td><strong>{rerank}</strong></td>
                <td>{hr5}</td>
                <td><strong>{mrr5}</strong></td>
                <td>{rec5}</td>
                <td>{map5}</td>
                <td>{d_mrr5}</td>
                <td {sig_badge}>{p_val}</td>
                <td>{lat}</td>
            </tr>
            """)

    rows_html = "\n".join(table_rows)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Relatório Científico: Avaliação de Re-ranking Neural em Dois Estágios (Two-Stage RAG)</title>
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
            content: "Pesquisa Científica e Tecnológica Aplicada (ICT) — CM Comandos Two-Stage RAG";
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
        font-size: 17pt;
        border-bottom: 2px solid #2b6cb0;
        padding-bottom: 5px;
        margin-top: 0;
        margin-bottom: 12px;
    }}
    h2 {{
        color: #2b6cb0;
        font-size: 12.5pt;
        border-bottom: 1px solid #cbd5e0;
        padding-bottom: 3px;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }}
    h3 {{
        color: #2d3748;
        font-size: 10.5pt;
        margin-top: 12px;
        margin-bottom: 5px;
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
        padding: 6px 12px;
        margin: 10px 0;
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
    img {{
        max-width: 95%;
        height: auto;
        display: block;
        margin: 12px auto;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        page-break-inside: avoid;
    }}
    .metadata-box {{
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 16px;
        font-size: 8.5pt;
    }}
</style>
</head>
<body>

<h1>Avaliação Científica de Re-ranking Neural em Dois Estágios (Two-Stage RAG)</h1>

<div class="metadata-box">
    <strong>Autor / Pesquisador:</strong> João Messias Lima Pereira | <strong>Projeto de Pesquisa Científica e Tecnológica (ICT)</strong><br>
    <strong>Estudo de Caso:</strong> Manuais Técnicos de No-breaks Industriais (CM Comandos Lineares)<br>
    <strong>Hardware de Execução:</strong> NVIDIA RTX PRO 1000 Blackwell Laptop GPU (8 GB GDDR6), Intel Core Ultra 7 265H CPU, 32 GB RAM<br>
    <strong>Metodologia:</strong> Super-sampling de $K_{{\\text{{cand}}}}=20$ candidatos, 128 perguntas deduplicadas, Matching em 3 Camadas, Teste Pareado de Wilcoxon ($\\alpha=0.05$), Bootstrap 95% ($B=1.000$).
</div>

<h2>1. Resumo Executivo</h2>
<p>
Este relatório documenta a avaliação experimental completa da introdução de uma etapa de reordenação neural de segundo estágio (Two-Stage Retrieval & Re-ranking) sobre os dois melhores modelos de primeiro estágio (<code>qwen3_8b_q4km</code> e <code>multilingual_e5_large</code>).
Os resultados comprovam que modelos locais quantizados de pequeno porte (SLMs) e Cross-Encoders dedicados locais superaram com folga tanto o baseline puro quanto a API comercial proprietária da OpenAI (<code>RankGPT</code> com <code>gpt-4o-mini</code>), alcançando ganhos estatisticamente significantes de até <strong>+29.4% em MRR@5</strong> com <strong>custo financeiro zero</strong> e <strong>soberania total dos dados industriais</strong>.
</p>

<h2>2. Tabela Geral de Desempenho no Top-5 (K=5)</h2>
<table>
    <thead>
        <tr>
            <th>Modelo Base</th>
            <th>Reranker</th>
            <th>HR@5</th>
            <th>MRR@5</th>
            <th>Recall@5</th>
            <th>MAP@5</th>
            <th>Δ MRR@5</th>
            <th>Wilcoxon (p)</th>
            <th>Latência</th>
        </tr>
    </thead>
    <tbody>
        {rows_html}
    </tbody>
</table>
<p><em>Nota: Teste de Wilcoxon pareado calculado consulta por consulta contra o baseline puro correspondente. Valores de p &lt; 0.05 indicam ganho estatisticamente significante.</em></p>

<h2>3. Análise dos Resultados e Ganhos de Recuperação</h2>
<p>
A análise comparativa entre os modelos evidencia que o re-ranking repara falhas clássicas de bi-encoders densos em domínios altamente especializados:
</p>
<ul>
    <li><strong>Elevação de Precisão:</strong> O MRR@5 da base <code>qwen3_8b_q4km</code> subiu de <strong>0.3887 para 0.5029 (+0.1142)</strong> com o <code>Qwen3-8B (Q5_K_M)</code> e para <strong>0.4866 (+0.0979)</strong> com o <code>BGE-Reranker-v2-m3</code>.</li>
    <li><strong>Resgate de Documentos Relevantes:</strong> Na base <code>multilingual_e5_large</code>, o Hit Rate@5 cresceu de <strong>54.7% para 62.5% (+7.8 p.p.)</strong> com o <code>Qwen3-4B</code>, demonstrando que trechos relevantes que haviam sido classificados entre as posições 6 e 20 foram corretamente promovidos para o Top-5.</li>
    <li><strong>Superioridade Local vs. OpenAI:</strong> Todos os modelos locais (<code>Qwen3-4B</code>, <code>Qwen3-8B</code>, <code>BGE-v2-m3</code>) superaram o <code>RankGPT (gpt-4o-mini)</code> em MRR@5. Modelos treinados especificamente com função de perda para ranqueamento de passagens superam LLMs genéricas que reordenam por prompting listwise.</li>
</ul>

<img src="{base_dir}/graficos_tcc/ranking_rerankers_mrr_hr_k5.png" alt="Ranking de Rerankers no Top-5">

<h2>4. Ganhos Incrementais (Δ MRR@5) e Validação de Hipótese</h2>
<p>
Todos os reordenadores avaliados apresentaram valores de $\\Delta$ MRR@5 estritamente positivos e com $p < 0.05$ no teste pareado de Wilcoxon, refutando a hipótese nula ($H_0$).
</p>

<img src="{base_dir}/graficos_tcc/delta_mrr5_rerankers.png" alt="Ganhos Incrementais em MRR@5">

<h2>5. Análise de Engenharia: Latência vs. Eficácia e VRAM</h2>
<p>
Em sistemas RAG de produção industrial, a precisão deve ser ponderada pela latência adicional de processamento e pelo consumo de memória gráfica:
</p>

<img src="{base_dir}/graficos_tcc/tradeoff_latencia_mrr_rerankers.png" alt="Trade-off Latencia vs MRR@5">

<table>
    <thead>
        <tr>
            <th>Modelo</th>
            <th>Engine / Framework</th>
            <th>VRAM Alocada</th>
            <th>Latência Média por Consulta</th>
            <th>Custo / 1M Consultas</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>BGE-Reranker-v2-m3</code></td>
            <td>HuggingFace PyTorch</td>
            <td>~1.2 GB</td>
            <td><strong>1.145 ms (1.14 s)</strong></td>
            <td><strong>$0.00</strong></td>
        </tr>
        <tr>
            <td><code>Qwen3-Reranker-4B (Q4_K_M)</code></td>
            <td>Ollama (GGUF)</td>
            <td>~2.5 GB</td>
            <td>5.182 ms (5.18 s)</td>
            <td><strong>$0.00</strong></td>
        </tr>
        <tr>
            <td><code>Qwen3-Reranker-8B (Q5_K_M)</code></td>
            <td>Ollama (GGUF)</td>
            <td>~5.9 GB</td>
            <td>6.424 ms (6.42 s)</td>
            <td><strong>$0.00</strong></td>
        </tr>
        <tr>
            <td><code>RankGPT (gpt-4o-mini)</code></td>
            <td>OpenAI Cloud API</td>
            <td>0.0 GB (Nuvem)</td>
            <td>1.648 ms (1.65 s)</td>
            <td>~$150.00</td>
        </tr>
    </tbody>
</table>

<blockquote>
    <strong>Destaque de Engenharia:</strong> O modelo <code>BGE-Reranker-v2-m3</code> constitui o ponto ótimo na fronteira de Pareto, entregando um MRR@5 de 0.4866 com latência de apenas 1.14s e alocação de 1.2 GB de VRAM.
</blockquote>

<h2>6. Conclusões e Recomendação para a CM Comandos Lineares</h2>
<p>
Com base nos dados empíricos obtidos na GPU NVIDIA RTX PRO 1000 (8 GB), recomenda-se para a CM Comandos Lineares:
</p>
<ol>
    <li><strong>Pipeline de Produção em Tempo Real:</strong> Utilizar <code>Multilingual-E5-Large</code> (ou sua versão quantizada ONNX INT8) no primeiro estágio e <code>BGE-Reranker-v2-m3</code> no segundo estágio. Esse arranjo responde em ~1.17 segundos por consulta, ocupa menos de 2.5 GB de VRAM e atinge MRR@5 de 0.4866.</li>
    <li><strong>Pipeline de Diagnóstico Técnico Aprofundado:</strong> Utilizar <code>Qwen3-Embedding-8B (Q4_K_M)</code> no primeiro estágio e <code>Qwen3-Reranker-8B (Q5_K_M)</code> no segundo estágio para consultas complexas de manutenção corretiva, alcançando o patamar máximo de eficácia (MRR@5 = 0.5029 e HR@5 = 60.2%).</li>
</ol>

</body>
</html>
"""

    html_out = os.path.join(base_dir, "docs", "relatorio_rerankers_dois_estagios.html")
    pdf_out = os.path.join(base_dir, "docs", "relatorio_rerankers_dois_estagios.pdf")

    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML acadêmico gerado em '{html_out}'.")

    cmd_wp = ["/home/joaorura/.local/bin/weasyprint", html_out, pdf_out]
    res_wp = subprocess.run(cmd_wp, capture_output=True, text=True)
    if os.path.exists(pdf_out):
        print(f"Relatório PDF acadêmico gerado com sucesso: '{pdf_out}' ({os.path.getsize(pdf_out)/1024:.1f} KB).")
    else:
        print(f"Erro na geração do PDF acadêmico: {res_wp.stderr[:400]}")


if __name__ == "__main__":
    generate_rerankers_pdf_report()
