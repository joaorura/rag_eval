"""Consolidação de métricas e testes de hipótese estatística para o benchmark de embeddings.

Calcula médias, intervalos de confiança via Bootstrap (95%) e teste pareado de Wilcoxon (OC-8),
exportando os resultados para CSV e Markdown.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

BASELINE_ID = "openai_3_small"


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 95.0) -> tuple[float, float]:
    """Calcula intervalo de confiança não-paramétrico por Bootstrap."""
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


def consolidate_results(results_dir: str = "results", output_csv: str = "results/consolidated_embeddings_metrics.csv"):
    json_files = glob.glob(os.path.join(results_dir, "raw_retrievals_*.json"))
    if not json_files:
        print(f"Nenhum arquivo de resultado encontrado em '{results_dir}'.")
        return

    models_data = {}
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            models_data[data["model_id"]] = data

    print(f"Modelos carregados para consolidação: {list(models_data.keys())}")

    # Baseline para comparação pareada (Wilcoxon)
    baseline_records = None
    if BASELINE_ID in models_data:
        baseline_records = {r["query_index"]: r for r in models_data[BASELINE_ID]["records"]}

    rows = []
    k_targets = [2, 5, 10]

    for model_id, m_content in models_data.items():
        records = m_content["records"]
        row: dict[str, Any] = {
            "model_id": model_id,
            "provider": m_content.get("config", {}).get("provider", "unknown"),
            "dimension": m_content.get("config", {}).get("dimension", 0),
            "quantization": m_content.get("config", {}).get("quantization", "FP16/FP32"),
            "total_queries": len(records),
            "mean_latency_ms": m_content.get("mean_latency_ms", 0.0),
            "index_build_time_s": m_content.get("index_build_time_s", 0.0),
        }

        # Extrai vetores de métricas por consulta
        metrics_by_name: dict[str, list[float]] = {}
        for r in records:
            for m_key, val in r.get("metrics", {}).items():
                metrics_by_name.setdefault(m_key, []).append(val)

        # Calcula média e CI para cada métrica
        for m_key, vals in metrics_by_name.items():
            mean_val = float(np.mean(vals))
            ci_low, ci_high = bootstrap_ci(vals)
            row[f"{m_key}_mean"] = round(mean_val, 4)
            row[f"{m_key}_ci95"] = f"[{ci_low:.3f}, {ci_high:.3f}]"

        # Teste estatístico de Wilcoxon vs Baseline para K=5
        if baseline_records and model_id != BASELINE_ID:
            base_mrr5 = []
            cand_mrr5 = []
            base_hr5 = []
            cand_hr5 = []

            for r in records:
                q_idx = r["query_index"]
                if q_idx in baseline_records:
                    base_r = baseline_records[q_idx]
                    base_mrr5.append(base_r["metrics"].get("mrr@5", 0.0))
                    cand_mrr5.append(r["metrics"].get("mrr@5", 0.0))
                    base_hr5.append(base_r["metrics"].get("hit_rate@5", 0.0))
                    cand_hr5.append(r["metrics"].get("hit_rate@5", 0.0))

            # Wilcoxon para MRR@5
            diff_mrr = np.array(cand_mrr5) - np.array(base_mrr5)
            if np.all(diff_mrr == 0):
                p_mrr = 1.0
            else:
                try:
                    res_mrr = stats.wilcoxon(cand_mrr5, base_mrr5, alternative="two-sided")
                    p_mrr = float(res_mrr.pvalue)
                except Exception:
                    p_mrr = 1.0

            # Wilcoxon para HitRate@5
            diff_hr = np.array(cand_hr5) - np.array(base_hr5)
            if np.all(diff_hr == 0):
                p_hr = 1.0
            else:
                try:
                    res_hr = stats.wilcoxon(cand_hr5, base_hr5, alternative="two-sided")
                    p_hr = float(res_hr.pvalue)
                except Exception:
                    p_hr = 1.0

            row["wilcoxon_p_mrr5_vs_baseline"] = round(p_mrr, 4)
            row["wilcoxon_p_hr5_vs_baseline"] = round(p_hr, 4)
        else:
            row["wilcoxon_p_mrr5_vs_baseline"] = "-"
            row["wilcoxon_p_hr5_vs_baseline"] = "-"

        rows.append(row)

    df = pd.DataFrame(rows)
    # Ordena pelo MRR@5 decrescente se disponível
    if "mrr@5_mean" in df.columns:
        df = df.sort_values(by="mrr@5_mean", ascending=False)

    df.to_csv(output_csv, index=False)
    print(f"Métricas consolidadas salvas em: {output_csv}")

    # Gera tabela Markdown simplificada para o relatório de pesquisa (ICT)
    display_cols = [
        "model_id", "quantization", "hit_rate@5_mean", "mrr@5_mean", "recall@5_mean",
        "map@5_mean", "mean_latency_ms", "wilcoxon_p_mrr5_vs_baseline"
    ]
    present_cols = [c for c in display_cols if c in df.columns]
    md_table = df[present_cols].to_markdown(index=False)

    table_path = os.path.join(results_dir, "summary_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Tabela Resumo: Desempenho Comparativo de Recuperação (K=5)\n\n")
        f.write(md_table)
        f.write("\n\n*Nota: Valores de p < 0.05 no teste de Wilcoxon indicam diferença estatisticamente significante em relação ao baseline da OpenAI.*\n")

    print("\n=== Resumo do Benchmark (K=5) ===")
    print(md_table)


if __name__ == "__main__":
    consolidate_results()
