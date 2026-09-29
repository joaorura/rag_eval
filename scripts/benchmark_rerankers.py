"""Benchmark de Rerankers (Fase 2) para Avaliação em Dois Estágios.

Avalia o ganho de recuperação ao aplicar diferentes arquiteturas de reranking sobre
candidatos gerados pelos melhores modelos de embedding da Fase 1 (qwen3_8b_q4km e multilingual_e5_large).

Implementa:
- Super-sampling com K_cand = 20 candidatos por consulta.
- Cache persistente de candidatos em disco para desassociação do índice vetorial.
- Gestão estrita de VRAM na GPU entre rerankers subsequentes.
- Avaliação de métricas de IR (Hit Rate, Recall, MRR, MAP) em K = [2, 5, 10].
- Compatibilidade com CLI e modo smoke test (--limit).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from typing import Any

# Garante a raiz do projeto no sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests
import torch
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from scripts.matching_engine import evaluate_query_retrieval, load_and_deduplicate_testset
from scripts.rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    NoneReranker,
    Qwen3OllamaReranker,
    RankGPTReranker,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fixação de Sementes para Reprodutibilidade Científica
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_MODELS_CONFIG = {
    "qwen3_8b_q4km": {
        "name": "qwen3-embedding:8b",
        "provider": "ollama",
        "storage_dir": "storage/index_qwen3_8b_q4km",
    },
    "multilingual_e5_large": {
        "name": "intfloat/multilingual-e5-large",
        "provider": "huggingface",
        "storage_dir": "storage/index_multilingual_e5_large",
        "use_e5_prefixes": True,
    },
}

RERANKER_MODELS = {
    "none": None,
    "qwen3_4b_q4km": "dengcao/Qwen3-Reranker-4B:Q4_K_M",
    "qwen3_8b_q5km": "dengcao/Qwen3-Reranker-8B:Q5_K_M",
    "bge_reranker_v2_m3": "BAAI/bge-reranker-v2-m3",
    "rankgpt": "gpt-4o-mini",
}


def clean_vram(reranker: Any | None = None, model_name: str | None = None) -> None:
    """Libera rigorosamente a memória VRAM da GPU e objetos de processo."""
    if reranker is not None and hasattr(reranker, "unload") and callable(reranker.unload):
        try:
            reranker.unload()
        except Exception as e:
            logger.warning(f"Erro ao descarregar reranker: {e}")

    # Descarrega modelo Ollama específico se fornecido
    if model_name:
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=5,
            )
        except Exception:
            pass

    # Descarrega proativamente quaisquer modelos Ollama conhecidos
    for m in [
        "qwen3-embedding:8b",
        "dengcao/Qwen3-Reranker-4B:Q4_K_M",
        "dengcao/Qwen3-Reranker-8B:Q5_K_M",
    ]:
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": m, "keep_alive": 0},
                timeout=3,
            )
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def instantiate_reranker(reranker_id: str) -> BaseReranker:
    """Instancia a classe de reranker correspondente ao identificador."""
    if reranker_id == "none":
        return NoneReranker()
    elif reranker_id == "qwen3_4b_q4km":
        return Qwen3OllamaReranker(model_name="dengcao/Qwen3-Reranker-4B:Q4_K_M")
    elif reranker_id == "qwen3_8b_q5km":
        return Qwen3OllamaReranker(model_name="dengcao/Qwen3-Reranker-8B:Q5_K_M")
    elif reranker_id == "bge_reranker_v2_m3":
        return CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    elif reranker_id == "rankgpt":
        return RankGPTReranker(model_name="gpt-4o-mini")
    else:
        raise ValueError(f"Reranker desconhecido: '{reranker_id}'")


def load_base_index_retriever(base_id: str, k_cand: int = 20):
    """Carrega o índice vetorial e cria o retriever base."""
    if base_id not in BASE_MODELS_CONFIG:
        raise ValueError(f"Modelo base '{base_id}' desconhecido. Opções: {list(BASE_MODELS_CONFIG.keys())}")

    cfg = BASE_MODELS_CONFIG[base_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    provider = cfg["provider"]
    model_name = cfg["name"]

    if provider == "ollama":
        embed_model = OllamaEmbedding(model_name=model_name, base_url="http://localhost:11434")
    elif provider == "huggingface":
        if cfg.get("use_e5_prefixes", False):
            embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                text_instruction="passage: ",
                query_instruction="query: ",
                embed_batch_size=32,
            )
        else:
            embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                embed_batch_size=32,
            )
    else:
        raise ValueError(f"Provedor desconhecido: {provider}")

    Settings.embed_model = embed_model
    storage_dir = cfg["storage_dir"]
    if not os.path.exists(storage_dir):
        raise FileNotFoundError(f"Diretório de índice '{storage_dir}' não encontrado para o modelo '{base_id}'.")

    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=k_cand)
    return retriever, embed_model, index


def get_or_create_candidates(
    base_id: str,
    k_cand: int = 20,
    testset_path: str = "testset_openai_4omini.jsonl",
    results_dir: str = "results",
) -> list[dict[str, Any]]:
    """Obtém os Top-K candidatos para todas as consultas do testset com cache em disco."""
    os.makedirs(results_dir, exist_ok=True)
    cache_path = os.path.join(results_dir, f"candidates_{base_id}_k{k_cand}.json")

    if os.path.exists(cache_path):
        print(f"[{base_id}] Encontrado cache de candidatos em '{cache_path}'. Carregando...")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("candidates") or data.get("records") or data
        print(f"[{base_id}] {len(records)} consultas carregadas com sucesso do cache.")
        return records

    print(f"[{base_id}] Cache não encontrado em '{cache_path}'.")
    print(f"[{base_id}] Carregando testset de '{testset_path}' para extração de Top-{k_cand} candidatos...")
    testset, total_orig, total_eff = load_and_deduplicate_testset(testset_path)
    print(f"[{base_id}] Testset deduplicado: {total_orig} -> {total_eff} consultas.")

    # Carrega modelo e índice vetorial
    retriever, embed_model, index = load_base_index_retriever(base_id, k_cand=k_cand)

    candidate_records: list[dict[str, Any]] = []
    t_start = time.time()

    try:
        for q_idx, item in enumerate(testset):
            query = item["user_input"]
            ref_contexts = item.get("reference_contexts", [])

            nodes = retriever.retrieve(query)
            cands_data = [
                {
                    "rank": idx + 1,
                    "node_id": n.node_id,
                    "text": n.text,
                    "score": float(n.score) if n.score is not None else 0.0,
                    "metadata": n.metadata or {},
                }
                for idx, n in enumerate(nodes)
            ]

            candidate_records.append(
                {
                    "query_index": q_idx,
                    "user_input": query,
                    "reference_contexts": ref_contexts,
                    "candidates": cands_data,
                }
            )

            if (q_idx + 1) % 25 == 0 or (q_idx + 1) == len(testset):
                elapsed = time.time() - t_start
                print(f"[{base_id}] Recuperados candidatos para {q_idx + 1}/{len(testset)} consultas ({elapsed:.1f}s)...")

        # Persiste o cache no disco
        payload = {
            "base_model": base_id,
            "top_k_candidates": k_cand,
            "total_queries": len(candidate_records),
            "records": candidate_records,
            "candidates": candidate_records,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[{base_id}] Candidatos salvos com sucesso em: {cache_path}")

    finally:
        # Liberação imediata da memória do retriever base
        clean_vram(model_name=BASE_MODELS_CONFIG[base_id]["name"])

    return candidate_records


def reconstruct_candidate_nodes(candidate_items: list[dict[str, Any]]) -> list[NodeWithScore]:
    """Reconstrói objetos NodeWithScore a partir dos dados serializados de candidatos."""
    reconstructed: list[NodeWithScore] = []
    for idx, c in enumerate(candidate_items):
        node = TextNode(
            text=c.get("text", ""),
            id_=c.get("node_id") or c.get("id_") or f"node_{idx}",
            metadata=c.get("metadata", {}),
        )
        score = float(c.get("score", 0.0))
        reconstructed.append(NodeWithScore(node=node, score=score))
    return reconstructed


def parse_list_arg(arg_list: list[str]) -> list[str]:
    """Suporta argumentos separados por espaço ou por vírgula no CLI."""
    flattened = []
    for item in arg_list:
        for piece in item.split(","):
            p = piece.strip()
            if p:
                flattened.append(p)
    return flattened


def run_reranker_benchmark(
    base_models: list[str],
    rerankers: list[str],
    top_k_candidates: int = 20,
    limit: int | None = None,
    testset_path: str = "testset_openai_4omini.jsonl",
    output_dir: str = "results",
) -> dict[str, Any]:
    """Executa a rotina completa de benchmark de rerankers."""
    load_dotenv()
    os.makedirs(output_dir, exist_ok=True)
    k_values = [2, 5, 10]
    max_k = max(k_values)

    summary_results: dict[str, Any] = {}

    for base_id in base_models:
        if base_id not in BASE_MODELS_CONFIG:
            print(f"Aviso: Modelo base '{base_id}' desconhecido. Pulando.")
            continue

        print("\n" + "=" * 70)
        print(f"  BASE RETRIEVER: {base_id}")
        print("=" * 70)

        # 1. Carrega ou extrai pool de candidatos
        all_candidate_records = get_or_create_candidates(
            base_id=base_id,
            k_cand=top_k_candidates,
            testset_path=testset_path,
            results_dir=output_dir,
        )

        # Aplica limitação se estiver em modo smoke test
        if limit and limit > 0:
            candidate_records = all_candidate_records[:limit]
            print(f"[{base_id}] Modo smoke test: avaliando primeiras {len(candidate_records)} consultas.")
        else:
            candidate_records = all_candidate_records

        # 2. Avaliação de cada reranker
        for reranker_id in rerankers:
            print("\n" + "-" * 50)
            print(f"  Avaliando Reranker: {reranker_id} sobre {base_id}")
            print("-" * 50)

            reranker = None
            try:
                reranker = instantiate_reranker(reranker_id)
                eval_records = []
                latencies = []

                for q_idx, item in enumerate(candidate_records):
                    query = item["user_input"]
                    ref_contexts = item.get("reference_contexts", [])
                    cands = item.get("candidates") or item.get("nodes") or []
                    candidate_nodes = reconstruct_candidate_nodes(cands)

                    # Rerank e medição de latência
                    t0 = time.time()
                    reranked_nodes = reranker.rerank(query, candidate_nodes, top_n=max_k)
                    lat_ms = (time.time() - t0) * 1000.0
                    latencies.append(lat_ms)

                    reranked_nodes_data = [
                        {
                            "rank": idx + 1,
                            "node_id": n.node.node_id if hasattr(n.node, "node_id") else getattr(n.node, "id_", f"node_{idx}"),
                            "text": n.node.get_content() if hasattr(n.node, "get_content") else getattr(n.node, "text", ""),
                            "score": float(n.score) if n.score is not None else 0.0,
                        }
                        for idx, n in enumerate(reranked_nodes)
                    ]

                    # Cálculo das métricas de IR
                    metrics = evaluate_query_retrieval(reranked_nodes_data, ref_contexts, k_values=k_values)

                    record = {
                        "query_index": item.get("query_index", q_idx),
                        "user_input": query,
                        "reference_contexts": ref_contexts,
                        "latency_ms": lat_ms,
                        "metrics": metrics,
                        "reranked_nodes": reranked_nodes_data,
                        "retrieved_nodes": reranked_nodes_data,
                    }
                    eval_records.append(record)

                    if (q_idx + 1) % 10 == 0 or (q_idx + 1) == len(candidate_records):
                        mean_so_far = np.mean(latencies)
                        print(f"[{reranker_id}] Processadas {q_idx + 1}/{len(candidate_records)} consultas (latência média: {mean_so_far:.1f} ms)...")

                mean_lat = float(np.mean(latencies)) if latencies else 0.0
                out_payload = {
                    "base_model": base_id,
                    "reranker": reranker_id,
                    "total_queries": len(eval_records),
                    "mean_latency_ms": mean_lat,
                    "records": eval_records,
                }

                out_filename = f"raw_rerank_{base_id}_{reranker_id}.json"
                out_filepath = os.path.join(output_dir, out_filename)
                with open(out_filepath, "w", encoding="utf-8") as f:
                    json.dump(out_payload, f, ensure_ascii=False, indent=2)

                print(f"[{reranker_id}] Resultados salvos com sucesso em: {out_filepath}")

                # Consolidação para sumário
                hr5_mean = float(np.mean([r["metrics"].get("hit_rate@5", 0.0) for r in eval_records]))
                rec5_mean = float(np.mean([r["metrics"].get("recall@5", 0.0) for r in eval_records]))
                mrr5_mean = float(np.mean([r["metrics"].get("mrr@5", 0.0) for r in eval_records]))
                map5_mean = float(np.mean([r["metrics"].get("map@5", 0.0) for r in eval_records]))

                summary_key = f"{base_id}__{reranker_id}"
                summary_results[summary_key] = {
                    "base_model": base_id,
                    "reranker": reranker_id,
                    "total_queries": len(eval_records),
                    "mean_latency_ms": mean_lat,
                    "hit_rate@5": hr5_mean,
                    "recall@5": rec5_mean,
                    "mrr@5": mrr5_mean,
                    "map@5": map5_mean,
                }

            finally:
                # Gestão estrita de VRAM entre rerankers subsequentes
                print(f"[{reranker_id}] Liberando VRAM e limpando caches...")
                clean_vram(reranker=reranker)

    # Imprime tabela consolidada final
    print("\n" + "=" * 90)
    print("  SUMÁRIO DOS RESULTADOS DO BENCHMARK DE RERANKERS")
    print("=" * 90)
    header = f"{'Base Retriever':<24} | {'Reranker':<20} | {'Lat (ms)':<10} | {'HR@5':<8} | {'Rec@5':<8} | {'MRR@5':<8} | {'MAP@5':<8}"
    print(header)
    print("-" * len(header))
    for k, s in summary_results.items():
        print(
            f"{s['base_model']:<24} | {s['reranker']:<20} | {s['mean_latency_ms']:<10.1f} | "
            f"{s['hit_rate@5']:<8.4f} | {s['recall@5']:<8.4f} | {s['mrr@5']:<8.4f} | {s['map@5']:<8.4f}"
        )
    print("=" * 90)

    return summary_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark de Rerankers (Fase 2)")
    parser.add_argument(
        "--base-models",
        nargs="+",
        default=["qwen3_8b_q4km", "multilingual_e5_large"],
        help="Modelos base de recuperação (default: qwen3_8b_q4km multilingual_e5_large)",
    )
    parser.add_argument(
        "--rerankers",
        nargs="+",
        default=["none", "qwen3_4b_q4km", "qwen3_8b_q5km", "bge_reranker_v2_m3", "rankgpt"],
        help="Modelos de reranking (default: none qwen3_4b_q4km qwen3_8b_q5km bge_reranker_v2_m3 rankgpt)",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=20,
        help="Super-sampling de candidatos pelo modelo base (default: 20)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de consultas para teste rápido / smoke test",
    )
    parser.add_argument(
        "--testset-path",
        type=str,
        default="testset_openai_4omini.jsonl",
        help="Caminho do arquivo com ground truth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Diretório para salvar os resultados",
    )

    args = parser.parse_args()
    base_models = parse_list_arg(args.base_models)
    rerankers = parse_list_arg(args.rerankers)

    run_reranker_benchmark(
        base_models=base_models,
        rerankers=rerankers,
        top_k_candidates=args.top_k_candidates,
        limit=args.limit,
        testset_path=args.testset_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
