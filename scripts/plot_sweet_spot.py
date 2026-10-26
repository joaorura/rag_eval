"""Gera gráfico comparativo de Custo vs. Qualidade (RAGAS) e destaca o Sweet Spot.

Compara GPT-6 Luna, GPT-6 Sol e baselines (GPT-4o, GPT-4o-mini).
Salva em 'graficos_tcc/pareto_sweet_spot_luna_sol.png' e 'docs/pareto_sweet_spot_luna_sol.png'.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_sweet_spot_plot():
    # 1. Carrega resultados do Luna
    luna_path = "results_luna_reasoning_efforts.json"
    sol_path = "results_sol_reasoning_efforts.json"

    if not os.path.exists(luna_path):
        print(f"Erro: {luna_path} não encontrado.")
        return

    with open(luna_path, "r", encoding="utf-8") as f:
        luna_data = json.load(f)

    sol_data = {}
    if os.path.exists(sol_path):
        with open(sol_path, "r", encoding="utf-8") as f:
            sol_data = json.load(f)

    # Custos calculados para 30.000 reqs/mês (~1.500 tokens input, ~250 tokens output base)
    # Preços:
    # Luna: Input $0.10/M, Output $0.50/M
    # Sol: Input $2.00/M, Output $10.00/M
    # 4o-mini: Input $0.15/M, Output $0.60/M
    # 4o: Input $2.50/M, Output $10.00/M

    cost_luna = {
        "none": 8.25,
        "low": 12.75,
        "medium": 23.25,
        "high": 45.75,
    }

    cost_sol = {
        "none": 165.00,
        "low": 174.00,
        "medium": 195.00,
        "high": 240.00,
    }

    # Baselines históricos/estimados
    models = []
    
    # Adiciona Luna
    for effort in ["none", "low", "medium", "high"]:
        if effort in luna_data:
            f_val = luna_data[effort]["faithfulness"]
            r_val = luna_data[effort]["answer_relevancy"]
            c_val = cost_luna.get(effort, 10.0)
            models.append({
                "label": f"GPT-6 Luna ({effort})",
                "family": "Luna",
                "effort": effort,
                "cost": c_val,
                "faithfulness": f_val,
                "relevancy": r_val,
                "composite": (f_val + r_val) / 2.0,
                "color": "#1f77b4" if effort != "high" and effort != "none" else ("#2ca02c" if effort == "none" else "#0055ff"),
                "marker": "o" if effort != "high" and effort != "none" else ("s" if effort == "none" else "*"),
                "size": 180 if effort in ("none", "high") else 120
            })

    # Adiciona Sol se disponível
    for effort in ["none", "low", "medium", "high"]:
        if effort in sol_data:
            f_val = sol_data[effort]["faithfulness"]
            r_val = sol_data[effort]["answer_relevancy"]
            c_val = cost_sol.get(effort, 180.0)
            models.append({
                "label": f"GPT-6 Sol ({effort})",
                "family": "Sol",
                "effort": effort,
                "cost": c_val,
                "faithfulness": f_val,
                "relevancy": r_val,
                "composite": (f_val + r_val) / 2.0,
                "color": "#ff7f0e",
                "marker": "^",
                "size": 130
            })

    # Adiciona Baselines
    models.append({
        "label": "GPT-4o-mini (baseline)",
        "family": "Baseline",
        "effort": "none",
        "cost": 11.25,
        "faithfulness": 0.3333,
        "relevancy": 0.7500,
        "composite": (0.3333 + 0.7500) / 2.0,
        "color": "#7f7f7f",
        "marker": "D",
        "size": 110
    })
    models.append({
        "label": "GPT-4o (baseline)",
        "family": "Baseline",
        "effort": "none",
        "cost": 187.50,
        "faithfulness": 0.3800,
        "relevancy": 0.7900,
        "composite": (0.3800 + 0.7900) / 2.0,
        "color": "#d62728",
        "marker": "X",
        "size": 140
    })

    # Configuração de estilo do Pyplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.28)

    # -------------------------------------------------------------
    # Gráfico 1: Trade-off Custo vs Answer Relevancy (Pareto / Sweet Spot)
    # -------------------------------------------------------------
    ax1.set_title("Trade-off: Custo Operacional vs. Relevância da Resposta (RAGAS)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Custo Mensal Estimado ($ USD / 30k reqs) [Escala Log]", fontsize=11, labelpad=8)
    ax1.set_ylabel("Answer Relevancy (RAGAS)", fontsize=11, labelpad=8)
    ax1.set_xscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    for m in models:
        ax1.scatter(m["cost"], m["relevancy"], color=m["color"], marker=m["marker"], s=m["size"], edgecolors="black", linewidth=1.2, zorder=4)
        
        # Posicionamento das legendas
        offset_x = 1.08
        offset_y = 0.003
        if "GPT-4o-mini" in m["label"]:
            offset_y = -0.015
        elif "none" in m["label"] and "Luna" in m["label"]:
            offset_y = 0.008
        elif "high" in m["label"] and "Luna" in m["label"]:
            offset_y = 0.008
        elif "GPT-4o (" in m["label"]:
            offset_y = -0.012

        ax1.annotate(
            m["label"],
            (m["cost"], m["relevancy"]),
            textcoords="offset points",
            xytext=(10, offset_y * 1000),
            fontsize=9,
            fontweight="bold" if "Luna" in m["label"] else "normal",
            alpha=0.9
        )

    # Destacar os Sweet Spots
    luna_none = next((m for m in models if m["label"] == "GPT-6 Luna (none)"), None)
    luna_high = next((m for m in models if m["label"] == "GPT-6 Luna (high)"), None)

    if luna_none:
        ax1.annotate(
            "★ SWEET SPOT OPERACIONAL\nMenor custo ($8.25/mês)\nAlta relevância sem reasoning",
            xy=(luna_none["cost"], luna_none["relevancy"]),
            xytext=(14, 0.72),
            arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=7),
            bbox=dict(boxstyle="round,pad=0.5", fc="#eafaf1", ec="#2ca02c", lw=1.5),
            fontsize=9,
            fontweight="bold",
            color="#196f3d"
        )

    if luna_high:
        ax1.annotate(
            "★ SWEET SPOT DE QUALIDADE\nMáxima Relevância (0.811)\n75% mais barato que GPT-4o",
            xy=(luna_high["cost"], luna_high["relevancy"]),
            xytext=(20, 0.835),
            arrowprops=dict(facecolor="#0055ff", shrink=0.08, width=1.5, headwidth=7),
            bbox=dict(boxstyle="round,pad=0.5", fc="#ebf5fb", ec="#0055ff", lw=1.5),
            fontsize=9,
            fontweight="bold",
            color="#1b4f72"
        )

    # -------------------------------------------------------------
    # Gráfico 2: Comparativo de Custo Mensal e Economia vs. Baseline
    # -------------------------------------------------------------
    ax2.set_title("Comparativo de Custo Mensal Estimado (30k consultas)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Custo Mensal ($ USD)", fontsize=11, labelpad=8)
    ax2.grid(True, axis="x", ls="--", alpha=0.5)

    # Ordena pelo custo decrescente
    sorted_models = sorted(models, key=lambda x: x["cost"])
    labels = [m["label"] for m in sorted_models]
    costs = [m["cost"] for m in sorted_models]
    colors = [m["color"] for m in sorted_models]

    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, costs, color=colors, edgecolor="black", height=0.65, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9.5)

    # Rótulo de valores nas barras
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 3, bar.get_y() + bar.get_height()/2, f"${w:.2f}", va="center", ha="left", fontsize=9, fontweight="bold")

    ax2.set_xlim(0, max(costs) * 1.15)

    os.makedirs("graficos_tcc", exist_ok=True)
    out1 = "graficos_tcc/pareto_sweet_spot_luna_sol.png"
    plt.savefig(out1, bbox_inches="tight")
    print(f"Gráfico salvo com sucesso em: {out1}")

if __name__ == "__main__":
    generate_sweet_spot_plot()
