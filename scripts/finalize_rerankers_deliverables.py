"""Script mestre para finalização de todos os entregáveis do Benchmark de Rerankers (Pesquisa ICT).

Executa em sequência:
1. Consolidação estatística dos 10 experimentos (scripts/consolidate_rerankers.py)
2. Geração de gráficos em 300 DPI em graficos_tcc/ (scripts/plot_reranker_graphs.py)
3. Construção e execução completa do Jupyter Notebook (scripts/generate_rerankers_notebook.py)
4. Compilação dos relatórios em PDF via WeasyPrint (scripts/generate_rerankers_pdf_report.py)
5. Geração da apresentação em Marp HTML/PDF e PowerPoint PPTX (scripts/generate_rerankers_presentation.py)
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from scripts.consolidate_rerankers import consolidate_rerankers

from scripts.plot_reranker_graphs import plot_reranker_graphs
from scripts.generate_rerankers_notebook import build_and_run_notebook
from scripts.generate_rerankers_pdf_report import generate_rerankers_pdf_report
from scripts.generate_rerankers_presentation import generate_rerankers_presentation


def finalize_all() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  INICIANDO CONSOLIDAÇÃO E GERAÇÃO DE ENTREGÁVEIS (TWO-STAGE RERANKERS)")
    print("=" * 70)

    # 1. Consolidação
    print("\n[Passo 1/5] Consolidando métricas e executando testes estatísticos...")
    df = consolidate_rerankers()
    print(f"Total de configurações consolidadas: {len(df)}")

    # 2. Gráficos
    print("\n[Passo 2/5] Gerando gráficos científicos em 300 DPI em 'graficos_tcc/'...")
    plot_reranker_graphs()

    # 3. Notebook
    print("\n[Passo 3/5] Construindo e executando Jupyter Notebook...")
    build_and_run_notebook()

    # 4. Apresentação (Marp HTML, PDF, PPTX)
    print("\n[Passo 4/5] Gerando apresentação (HTML, PDF, PPTX)...")
    generate_rerankers_presentation()

    # 5. Relatórios em PDF
    print("\n[Passo 5/5] Compilando relatórios PDF via WeasyPrint...")
    generate_rerankers_pdf_report()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  TODOS OS ENTREGÁVEIS GERADOS COM SUCESSO EM {elapsed:.1f}s!")
    print("=" * 70)

    # Listagem de verificação dos arquivos gerados
    deliverables = [
        "results/consolidated_reranker_metrics.csv",
        "results/summary_table_rerankers.md",
        "graficos_tcc/ranking_rerankers_mrr_hr_k5.png",
        "graficos_tcc/delta_mrr5_rerankers.png",
        "graficos_tcc/tradeoff_latencia_mrr_rerankers.png",
        "notebooks/avaliacao_rerankers_dois_estagios.ipynb",
        "docs/avaliacao_rerankers_dois_estagios.pdf",
        "docs/relatorio_rerankers_dois_estagios.pdf",
        "docs/apresentacao_rerankers.md",
        "docs/apresentacao_rerankers.html",
        "docs/apresentacao_rerankers.pdf",
        "docs/apresentacao_rerankers.pptx",
    ]

    print("\nStatus dos Arquivos:")
    for path in deliverables:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  [OK] {path} ({size_kb:.1f} KB)")
        else:
            print(f"  [PENDENTE] {path}")


if __name__ == "__main__":
    finalize_all()
