"""Consolidação estatística e análise de ganho para o benchmark de Re-ranking.

Calcula métricas agregadas (médias, IC 95% Bootstrap), ganhos incrementais (Δ),
e testes pareados de Wilcoxon contra o baseline puro (sem reranker).
Gera CSV consolidado e tabela resumo em Markdown.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 95.0) -> tuple[float, float]:
    """Calcula intervalo de confiança não-paramétrico via Bootstrap."""
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.array(values)
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (100.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha))
    high = float(np.percentile(boot_means, 100.0 - alpha))
    return low, high


def consolidate_rerankers(
    results_dir: str = "results",
    output_csv: str = "results/consolidated_reranker_metrics.csv",
    output_md: str = "results/summary_table_rerankers.md",
) -> pd.DataFrame:
    """Consolida os resultados de todos os benchmarks de reranking."""
    json_files = glob.glob(os.path.join(results_dir, "raw_rerank_*.json"))
    if not json_files:
        print(f"Nenhum arquivo 'raw_rerank_*.json' encontrado em '{results_dir}'.")
        return pd.DataFrame()

    runs_data: dict[tuple[str, str], dict[str, Any]] = {}
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            base_model = data.get("base_model", "unknown")
            reranker = data.get("reranker", "unknown")
            runs_data[(base_model, reranker)] = data

    print(f"Execuções carregadas para consolidação: {list(runs_data.keys())}")

    # Identifica todos os base_models
    base_models = sorted(list(set(b for b, _ in runs_data.keys())))
    rows = []

    for base_model in base_models:
        # Busca o baseline puro ('none') para este base_model
        baseline_key = (base_model, "none")
        baseline_data = runs_data.get(baseline_key)
        baseline_query_map = {}
        if baseline_data:
            baseline_query_map = {r["query_index"]: r for r in baseline_data.get("records", [])}

        # Ordena rerankers mantendo 'none' em primeiro
        model_rerankers = sorted(
            [r for b, r in runs_data.keys() if b == base_model],
            key=lambda x: (0 if x == "none" else 1, x),
        )

        for reranker in model_rerankers:
            data = runs_data[(base_model, reranker)]
            records = data.get("records", [])
            total_queries = len(records)
            if total_queries < 100:
                print(f"Pulando {base_model} + {reranker} pois é smoke test (apenas {total_queries} consultas).")
                continue
            mean_lat = data.get("mean_latency_ms", 0.0)

            row: dict[str, Any] = {
                "base_model": base_model,
                "reranker": reranker,
                "total_queries": total_queries,
                "mean_latency_ms": round(mean_lat, 2),
            }

            # Extrai vetores de métricas
            metrics_dict: dict[str, list[float]] = {}
            for r in records:
                for k, v in r.get("metrics", {}).items():
                    metrics_dict.setdefault(k, []).append(float(v))

            for m_key, vals in metrics_dict.items():
                m_mean = float(np.mean(vals))
                ci_low, ci_high = bootstrap_ci(vals)
                row[f"{m_key}_mean"] = round(m_mean, 4)
                row[f"{m_key}_ci95"] = f"[{ci_low:.3f}, {ci_high:.3f}]"

            # Estatísticas comparativas com o baseline 'none'
            if baseline_query_map and reranker != "none":
                cand_mrr5 = []
                base_mrr5 = []
                cand_hr5 = []
                base_hr5 = []

                for r in records:
                    q_idx = r["query_index"]
                    if q_idx in baseline_query_map:
                        base_rec = baseline_query_map[q_idx]
                        cand_mrr5.append(r.get("metrics", {}).get("mrr@5", 0.0))
                        base_mrr5.append(base_rec.get("metrics", {}).get("mrr@5", 0.0))
                        cand_hr5.append(r.get("metrics", {}).get("hit_rate@5", 0.0))
                        base_hr5.append(base_rec.get("metrics", {}).get("hit_rate@5", 0.0))

                delta_mrr5 = np.mean(cand_mrr5) - np.mean(base_mrr5) if cand_mrr5 else 0.0
                delta_hr5 = np.mean(cand_hr5) - np.mean(base_hr5) if cand_hr5 else 0.0

                row["delta_mrr@5"] = round(float(delta_mrr5), 4)
                row["delta_hit_rate@5"] = round(float(delta_hr5), 4)

                # Wilcoxon MRR@5
                diff_mrr = np.array(cand_mrr5) - np.array(base_mrr5)
                if np.all(diff_mrr == 0):
                    p_val_mrr = 1.0
                else:
                    try:
                        _, p_val_mrr = stats.wilcoxon(diff_mrr, zero_method="pratt")
                    except Exception:
                        p_val_mrr = 1.0
                row["p_value_wilcoxon_mrr5"] = round(float(p_val_mrr), 5)
                row["sig_mrr5"] = "Sim (p<0.05)" if p_val_mrr < 0.05 else "Não (p>=0.05)"

                # Wilcoxon HR@5
                diff_hr = np.array(cand_hr5) - np.array(base_hr5)
                if np.all(diff_hr == 0):
                    p_val_hr = 1.0
                else:
                    try:
                        _, p_val_hr = stats.wilcoxon(diff_hr, zero_method="pratt")
                    except Exception:
                        p_val_hr = 1.0
                row["p_value_wilcoxon_hr5"] = round(float(p_val_hr), 5)
            else:
                row["delta_mrr@5"] = 0.0
                row["delta_hit_rate@5"] = 0.0
                row["p_value_wilcoxon_mrr5"] = 1.0
                row["sig_mrr5"] = "Baseline"
                row["p_value_wilcoxon_hr5"] = 1.0

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Métricas consolidadas exportadas para '{output_csv}'.")

    # Gera Markdown Summary Table
    md_content = generate_markdown_summary(df)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Resumo em Markdown exportado para '{output_md}'.")

    return df


def generate_markdown_summary(df: pd.DataFrame) -> str:
    """Gera tabela comparativa limpa em Markdown."""
    lines = [
        "# Tabela Resumo do Benchmark de Re-ranking Neural (Two-Stage RAG)",
        "",
        "Avaliação do impacto de reordenadores neurais nos Top-20 candidatos recuperados pelos melhores modelos de embedding.",
        "",
        "| Modelo Base | Reranker | HR@5 | MRR@5 | Δ MRR@5 | Δ HR@5 | Wilcoxon (p) | Latência Re-rank (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in df.iterrows():
        base = row["base_model"]
        rerank = row["reranker"]
        hr5 = f"{row.get('hit_rate@5_mean', 0.0) * 100:.1f}%"
        mrr5 = f"{row.get('mrr@5_mean', 0.0):.4f}"
        d_mrr5 = f"{row.get('delta_mrr@5', 0.0):+.4f}" if rerank != "none" else "-"
        d_hr5 = f"{row.get('delta_hit_rate@5', 0.0) * 100:+.1f}%" if rerank != "none" else "-"
        p_val = f"{row.get('p_value_wilcoxon_mrr5', 1.0):.4f}" if rerank != "none" else "Baseline"
        lat = f"{row.get('mean_latency_ms', 0.0):.1f} ms"
        lines.append(f"| `{base}` | `{rerank}` | **{hr5}** | **{mrr5}** | {d_mrr5} | {d_hr5} | {p_val} | {lat} |")

    lines.append("")
    lines.append("*Nota: Teste de Wilcoxon pareado calculado sobre o ganho de MRR@5 consulta por consulta contra o Baseline Puro (`none`).*")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    consolidate_rerankers()
