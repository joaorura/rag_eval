"""Geração de gráficos científicos para o benchmark de Re-ranking Neural (Two-Stage RAG).

Gera 3 gráficos salvos em 'graficos_tcc/':
1. ranking_rerankers_mrr_hr_k5.png: Comparação de MRR@5 e HR@5 por Reranker
2. delta_mrr5_rerankers.png: Ganho incremental (Δ MRR@5) em relação ao baseline puro
3. tradeoff_latencia_mrr_rerankers.png: Trade-off Latência incremental (ms) vs. MRR@5
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

RERANKER_LABELS = {
    "none": "Puro (Sem Re-ranking)",
    "qwen3_4b_q4km": "Qwen3-Reranker-4B (Q4_K_M)",
    "qwen3_8b_q5km": "Qwen3-Reranker-8B (Q5_K_M)",
    "bge_reranker_v2_m3": "BGE-Reranker-v2-m3 (HF)",
    "rankgpt": "RankGPT (gpt-4o-mini)",
}

BASE_LABELS = {
    "qwen3_8b_q4km": "Base: Qwen3-Embedding-8B (Q4_K_M)",
    "multilingual_e5_large": "Base: Multilingual-E5-Large (ONNX INT8 / HF)",
}


def plot_reranker_graphs(
    csv_path: str = "results/consolidated_reranker_metrics.csv",
    output_dir: str = "graficos_tcc",
):
    if not os.path.exists(csv_path):
        print(f"Arquivo CSV '{csv_path}' não encontrado. Execute a consolidação primeiro.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1. Gráfico de Barras Agrupadas: MRR@5 e HR@5 por Modelo Base e Reranker
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    base_models = df["base_model"].unique()

    for idx, base in enumerate(base_models):
        ax = axes[idx]
        sub_df = df[df["base_model"] == base].copy()
        sub_df["label"] = sub_df["reranker"].map(lambda x: RERANKER_LABELS.get(x, x))
        sub_df = sub_df.sort_values(by="mrr@5_mean", ascending=True)

        y_pos = np.arange(len(sub_df))
        height = 0.35

        bars_mrr = ax.barh(y_pos - height / 2, sub_df["mrr@5_mean"], height, label="MRR@5", color="#1f77b4")
        bars_hr = ax.barh(y_pos + height / 2, sub_df["hit_rate@5_mean"], height, label="Hit Rate@5", color="#2ca02c")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_df["label"], fontsize=10)
        ax.set_xlabel("Pontuação Média (0 a 1.0)")
        ax.set_title(BASE_LABELS.get(base, base), fontsize=12, weight="bold")
        ax.set_xlim(0, 1.05)
        ax.legend(loc="lower right")

        for bar in bars_mrr:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)
        for bar in bars_hr:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    plt.suptitle("Impacto do Re-ranking no Top-5 (MRR@5 vs. Hit Rate@5)", fontsize=14, weight="bold")
    fpath1 = os.path.join(output_dir, "ranking_rerankers_mrr_hr_k5.png")
    plt.savefig(fpath1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico 1 salvo: {fpath1}")

    # 2. Gráfico de Barras: Ganho Incremental Δ MRR@5
    delta_df = df[df["reranker"] != "none"].copy()
    if not delta_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        delta_df["full_label"] = (
            delta_df["base_model"].map(lambda x: "Qwen3-8B" if "qwen" in x else "E5-Large")
            + " + "
            + delta_df["reranker"].map(lambda x: RERANKER_LABELS.get(x, x))
        )
        delta_df = delta_df.sort_values(by="delta_mrr@5", ascending=True)

        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in delta_df["delta_mrr@5"]]
        y_pos = np.arange(len(delta_df))

        bars = ax.barh(y_pos, delta_df["delta_mrr@5"], color=colors, height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(delta_df["full_label"], fontsize=10)
        ax.set_xlabel("Ganho Incremental em MRR@5 (Δ MRR@5)")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title("Ganho Incremental de MRR@5 proporcionado pelos Reordenadores Neurais", fontsize=13, weight="bold")

        for bar in bars:
            w = bar.get_width()
            offset = 0.005 if w >= 0 else -0.015
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.4f}", va="center", fontsize=9, weight="bold")

        fpath2 = os.path.join(output_dir, "delta_mrr5_rerankers.png")
        plt.savefig(fpath2, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Gráfico 2 salvo: {fpath2}")

    # 3. Trade-off: Latência incremental (ms) vs. MRR@5
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"none": "o", "qwen3_4b_q4km": "s", "qwen3_8b_q5km": "^", "bge_reranker_v2_m3": "D", "rankgpt": "*"}
    colors_base = {"qwen3_8b_q4km": "#1f77b4", "multilingual_e5_large": "#ff7f0e"}

    for _, row in df.iterrows():
        base = row["base_model"]
        rerank = row["reranker"]
        x = row["mean_latency_ms"]
        y = row["mrr@5_mean"]
        c = colors_base.get(base, "#333333")
        m = markers.get(rerank, "o")
        lbl = f"{'Qwen3' if 'qwen' in base else 'E5'} + {rerank}"

        ax.scatter(x, y, color=c, marker=m, s=140 if rerank == "rankgpt" else 90, alpha=0.9)
        ax.annotate(lbl, (x + 10, y + 0.003), fontsize=9)

    ax.set_xlabel("Latência Incremental por Consulta (ms)")
    ax.set_ylabel("MRR@5 Médio")
    ax.set_title("Trade-off de Engenharia: Eficácia (MRR@5) vs. Custo Computacional (Latência)", fontsize=13, weight="bold")
    fpath3 = os.path.join(output_dir, "tradeoff_latencia_mrr_rerankers.png")
    plt.savefig(fpath3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico 3 salvo: {fpath3}")


if __name__ == "__main__":
    plot_reranker_graphs()
