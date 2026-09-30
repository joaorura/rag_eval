"""Geração de gráficos científicos e figuras de publicação para o Projeto de Pesquisa (ICT).

Gera 3 gráficos salvos em 'graficos_tcc/':
1. ranking_hitrate_mrr_k5.png
2. curva_recuperacao_topk.png
3. tradeoff_latencia_mrr.png
"""

from __future__ import annotations

import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 12, "figure.autolayout": True})


def plot_graphs(csv_path: str = "results/consolidated_embeddings_metrics.csv", output_dir: str = "graficos_tcc"):
    if not os.path.exists(csv_path):
        print(f"Arquivo CSV não encontrado em '{csv_path}'. Execute a consolidação primeiro.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1. Gráfico de Barras: Ranking por MRR@5 e Hit Rate@5
    if "mrr@5_mean" in df.columns and "hit_rate@5_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sorted = df.sort_values(by="mrr@5_mean", ascending=True)

        y_positions = np.arange(len(df_sorted))
        height = 0.35

        bars_mrr = ax.barh(y_positions - height/2, df_sorted["mrr@5_mean"], height, label="MRR@5", color="#1f77b4")
        bars_hr = ax.barh(y_positions + height/2, df_sorted["hit_rate@5_mean"], height, label="Hit Rate@5", color="#2ca02c")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(df_sorted["model_id"])
        ax.set_xlabel("Pontuação Média (0 a 1.0)")
        ax.set_title("Comparação de Recuperação no Top-5: MRR@5 vs. Hit Rate@5", fontsize=14, weight="bold")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1.05)

        # Adiciona rótulos nos valores das barras
        for bar in bars_mrr:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=9)
        for bar in bars_hr:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=9)

        fpath = os.path.join(output_dir, "ranking_hitrate_mrr_k5.png")
        plt.savefig(fpath, dpi=300)
        plt.close()
        print(f"Gráfico 1 salvo em: {fpath}")

    # 2. Curva de Evolução do Recall por Top-K (K=2, 5, 10)
    k_cols = [c for c in ["recall@2_mean", "recall@5_mean", "recall@10_mean"] if c in df.columns]
    if len(k_cols) == 3:
        fig, ax = plt.subplots(figsize=(9, 6))
        k_vals = [2, 5, 10]

        for _, row in df.iterrows():
            rec_vals = [row["recall@2_mean"], row["recall@5_mean"], row["recall@10_mean"]]
            is_base = "openai" in str(row["model_id"])
            ax.plot(
                k_vals,
                rec_vals,
                marker="o",
                linewidth=2.5 if is_base else 1.8,
                linestyle="--" if is_base else "-",
                label=f"{row['model_id']}" + (" (Baseline)" if is_base else ""),
            )

        ax.set_xticks(k_vals)
        ax.set_xlabel("Valor de Top-K Recuperado")
        ax.set_ylabel("Context Recall Médio")
        ax.set_title("Curva de Recuperação de Contextos Relevantes por Top-K", fontsize=14, weight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0, 1.05)

        fpath = os.path.join(output_dir, "curva_recuperacao_topk.png")
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Gráfico 2 salvo em: {fpath}")

    # 3. Gráfico de Dispersão: Trade-off Eficiência (Latência) vs. Eficácia (MRR@5)
    if "mean_latency_ms" in df.columns and "mrr@5_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for _, row in df.iterrows():
            x = row["mean_latency_ms"]
            y = row["mrr@5_mean"]
            m_label = row["model_id"]
            is_q4 = "q4km" in str(m_label)
            is_base = "openai" in str(m_label)

            color = "#d62728" if is_base else ("#2ca02c" if is_q4 else "#1f77b4")
            ax.scatter(x, y, s=150, color=color, alpha=0.85, edgecolors="black", zorder=3)
            ax.text(x + 1.5, y + 0.005, m_label, fontsize=10, weight="bold" if (is_q4 or is_base) else "normal")

        ax.set_xlabel("Latência Média por Consulta (ms)")
        ax.set_ylabel("Eficácia de Recuperação (MRR@5)")
        ax.set_title("Trade-off de Engenharia: Eficiência (ms) vs. Eficácia (MRR@5)", fontsize=14, weight="bold")

        fpath = os.path.join(output_dir, "tradeoff_latencia_mrr.png")
        plt.savefig(fpath, dpi=300)
        plt.close()
        print(f"Gráfico 3 salvo em: {fpath}")


if __name__ == "__main__":
    plot_graphs()
