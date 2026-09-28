"""Módulo de casamento e cálculo de métricas de Information Retrieval (IR).

Implementa deduplicação do testset de ground truth e matching em 3 camadas:
1. Contenção estrita de substring.
2. Similaridade Fuzzy por conjuntos de tokens (RapidFuzz >= 75%).
3. Sobreposição lexical ROUGE-L (F1 >= 0.50).
"""

from __future__ import annotations

import json
from typing import Any
import pandas as pd
from rapidfuzz import fuzz
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def load_and_deduplicate_testset(jsonl_path: str) -> tuple[list[dict[str, Any]], int, int]:
    """Carrega o arquivo JSONL de ground truth e remove duplicatas.

    Retorna:
        tuple (lista_deduplicada, total_original, total_efetivo)
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        raw_items = [json.loads(line) for line in f if line.strip()]

    total_original = len(raw_items)
    seen = set()
    dedup_items = []

    for item in raw_items:
        # Chave composta por pergunta e tupla de contextos de referência
        ref_tuple = tuple(item.get('reference_contexts', []))
        key = (item.get('user_input', '').strip(), ref_tuple)
        if key not in seen:
            seen.add(key)
            dedup_items.append(item)

    total_efetivo = len(dedup_items)
    return dedup_items, total_original, total_efetivo


def is_context_match(node_text: str, ref_text: str, fuzzy_threshold: float = 85.0, rouge_threshold: float = 0.50) -> tuple[bool, str]:
    """Verifica se um nó recuperado corresponde a um trecho de referência.

    Normaliza espaços em branco e aplica 3 critérios (OR):
    1. Contenção de substring para trechos com >= 40 chars.
    2. RapidFuzz partial_ratio >= 80% ou token_set_ratio >= fuzzy_threshold (85%).
    3. ROUGE-L F1 >= rouge_threshold (0.50).
    """
    n_clean = " ".join(node_text.strip().lower().split())
    r_clean = " ".join(ref_text.strip().lower().split())

    if not n_clean or not r_clean:
        return False, "empty"

    # 1. Contenção direta normalizada
    if len(r_clean) >= 40 and (r_clean in n_clean or n_clean in r_clean):
        return True, "substring"

    # 2. RapidFuzz
    if len(r_clean) >= 30:
        p_ratio = fuzz.partial_ratio(r_clean, n_clean)
        if p_ratio >= 85.0:
            return True, f"partial_{p_ratio:.1f}"

    set_ratio = fuzz.token_set_ratio(n_clean, r_clean)
    if set_ratio >= fuzzy_threshold:
        return True, f"token_set_{set_ratio:.1f}"

    # 3. ROUGE-L Overlap
    rouge_res = scorer.score(r_clean, n_clean)
    f1_score = rouge_res['rougeL'].fmeasure
    if f1_score >= rouge_threshold:
        return True, f"rougeL_{f1_score:.2f}"

    return False, "none"


def evaluate_query_retrieval(
    retrieved_nodes: list[dict[str, Any]],
    reference_contexts: list[str],
    k_values: list[int] = [2, 5, 10]
) -> dict[str, Any]:
    """Calcula métricas de IR para uma única consulta em múltiplos valores de K.

    Métricas calculadas:
    - hit_rate@K
    - recall@K
    - mrr@K
    - map@K
    """
    results: dict[str, Any] = {}
    total_refs = len(reference_contexts)
    if total_refs == 0:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ['hit_rate', 'recall', 'mrr', 'map']}

    # Avalia a relevância de cada nó recuperado
    node_matches = []
    matched_refs_global = set()

    for idx, node in enumerate(retrieved_nodes):
        node_text = node.get("text", "")
        matched_ref_indices = []
        for r_idx, ref in enumerate(reference_contexts):
            matched, method = is_context_match(node_text, ref)
            if matched:
                matched_ref_indices.append(r_idx)
                matched_refs_global.add(r_idx)

        is_rel = len(matched_ref_indices) > 0
        node_matches.append({
            "rank": idx + 1,
            "is_relevant": is_rel,
            "matched_refs": matched_ref_indices
        })

    # Calcula as métricas para cada K
    for k in k_values:
        sub_nodes = node_matches[:k]
        
        # 1. Hit Rate@K
        hit = 1.0 if any(n["is_relevant"] for n in sub_nodes) else 0.0
        results[f"hit_rate@{k}"] = hit

        # 2. Context Recall@K (proporção de referências cobertas no Top-K)
        k_matched_refs = set()
        for n in sub_nodes:
            k_matched_refs.update(n["matched_refs"])
        recall = len(k_matched_refs) / total_refs
        results[f"recall@{k}"] = recall

        # 3. MRR@K
        first_rel_rank = next((n["rank"] for n in sub_nodes if n["is_relevant"]), None)
        mrr = (1.0 / first_rel_rank) if first_rel_rank else 0.0
        results[f"mrr@{k}"] = mrr

        # 4. MAP@K
        rel_count = 0
        prec_sum = 0.0
        for i, n in enumerate(sub_nodes):
            if n["is_relevant"]:
                rel_count += 1
                prec_sum += rel_count / (i + 1)
        map_val = prec_sum / min(total_refs, k) if total_refs > 0 else 0.0
        results[f"map@{k}"] = map_val

    return results
