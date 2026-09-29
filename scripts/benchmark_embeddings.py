"""Script principal de benchmark de recuperação para múltiplos modelos de embedding.

Executa avaliação Retriever-only contra ground truth, com isolamento de índices em disco,
gestão sequencial de VRAM na GPU e persistência dos resultados brutos.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from typing import Any

# Garante que a raiz do repositório esteja no sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests
import torch
from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from scripts.matching_engine import evaluate_query_retrieval, load_and_deduplicate_testset

# Fixação de Sementes para Reprodutibilidade (OC-9)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODELS_CONFIG = {
    "openai_3_small": {
        "name": "text-embedding-3-small",
        "provider": "openai",
        "dimension": 1536,
        "is_baseline": True,
    },
    "qwen3_8b_q4km": {
        "name": "qwen3-embedding:8b",
        "provider": "ollama",
        "dimension": 4096,
        "quantization": "Q4_K_M",
    },
    "qwen3_4b_q4km": {
        "name": "qwen3-embedding:4b",
        "provider": "ollama",
        "dimension": 2560,
        "quantization": "Q4_K_M",
    },
    "bge_m3": {
        "name": "bge-m3",
        "provider": "ollama",
        "dimension": 1024,
    },
    "nomic_embed_text": {
        "name": "nomic-embed-text",
        "provider": "ollama",
        "dimension": 768,
    },
    "multilingual_e5_large": {
        "name": "intfloat/multilingual-e5-large",
        "provider": "huggingface",
        "dimension": 1024,
        "use_e5_prefixes": True,
    },
    "bertimbau_sts": {
        "name": "juridics/bertimbau-base-portuguese-sts-scale",
        "provider": "huggingface",
        "dimension": 768,
    },
}


def unload_vram(model_cfg: dict[str, Any]) -> None:
    """Libera a VRAM da GPU entre avaliações sequenciais (OC-5)."""
    if model_cfg.get("provider") == "ollama":
        try:
            # Descarrega explicitamente o modelo no Ollama
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_cfg["name"], "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_embedding_instance(model_id: str, cfg: dict[str, Any]):
    """Instancia o modelo com aceleração CUDA e tratamento de prefixos."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    provider = cfg["provider"]
    model_name = cfg["name"]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env!")
        return OpenAIEmbedding(model=model_name, api_key=api_key)

    elif provider == "ollama":
        return OllamaEmbedding(model_name=model_name, base_url="http://localhost:11434")

    elif provider == "huggingface":
        if cfg.get("use_e5_prefixes", False):
            # Tratamento mandatória de prefixos para o E5 (OC-6)
            return HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                text_instruction="passage: ",
                query_instruction="query: ",
                embed_batch_size=32,
            )
        else:
            return HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                embed_batch_size=32,
            )

    raise ValueError(f"Provedor desconhecido: {provider}")


def get_or_create_index(model_id: str, cfg: dict[str, Any], data_path: str = "data", storage_base: str = "storage"):
    """Garante isolamento de persistência de cada índice vetorial."""
    persist_dir = os.path.join(storage_base, f"index_{model_id}")
    embed_model = get_embedding_instance(model_id, cfg)
    Settings.embed_model = embed_model

    if os.path.exists(persist_dir):
        print(f"[{model_id}] Carregando índice existente de '{persist_dir}'...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        index_build_time = 0.0
    else:
        print(f"[{model_id}] Construindo novo índice a partir de '{data_path}'...")
        t0 = time.time()
        documents = SimpleDirectoryReader(data_path).load_data()
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        index_build_time = time.time() - t0
        print(f"[{model_id}] Índice salvo em '{persist_dir}' em {index_build_time:.2f}s.")

    return index, index_build_time


def run_benchmark(
    selected_models: list[str],
    testset_path: str = "testset_openai_4omini.jsonl",
    limit: int | None = None,
    output_dir: str = "results",
):
    """Executa o benchmark sequencial para todos os modelos selecionados."""
    load_dotenv()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Carrega e deduplica o testset (OC-2)
    testset, total_orig, total_eff = load_and_deduplicate_testset(testset_path)
    print(f"Testset carregado: {total_orig} amostras originais -> {total_eff} amostras efetivas deduplicadas.")

    if limit and limit > 0:
        testset = testset[:limit]
        print(f"Modo Smoke Test ativado: executando para as primeiras {len(testset)} consultas.")

    for model_id in selected_models:
        if model_id not in MODELS_CONFIG:
            print(f"Aviso: Modelo '{model_id}' não reconhecido. Pulando.")
            continue

        cfg = MODELS_CONFIG[model_id]
        print(f"\n=======================================================")
        print(f"  Iniciando Avaliação: {model_id} ({cfg['name']})")
        print(f"=======================================================")

        try:
            index, index_build_time = get_or_create_index(model_id, cfg)
            retriever = index.as_retriever(similarity_top_k=10)

            retrieval_records = []
            latencies = []

            for q_idx, item in enumerate(testset):
                query = item["user_input"]
                ref_contexts = item.get("reference_contexts", [])

                t_start = time.time()
                retrieved_nodes = retriever.retrieve(query)
                lat = (time.time() - t_start) * 1000.0  # ms
                latencies.append(lat)

                nodes_data = [
                    {
                        "rank": idx + 1,
                        "node_id": node.node_id,
                        "text": node.text,
                        "score": float(node.score) if node.score is not None else 0.0,
                    }
                    for idx, node in enumerate(retrieved_nodes)
                ]

                # Avalia métricas de IR para a consulta
                metrics = evaluate_query_retrieval(nodes_data, ref_contexts, k_values=[2, 5, 10])

                record = {
                    "query_index": q_idx,
                    "user_input": query,
                    "reference_contexts": ref_contexts,
                    "latency_ms": lat,
                    "metrics": metrics,
                    "retrieved_nodes": nodes_data,
                }
                retrieval_records.append(record)

                if (q_idx + 1) % 25 == 0 or (q_idx + 1) == len(testset):
                    print(f"[{model_id}] Processadas {q_idx + 1}/{len(testset)} consultas (latência média: {np.mean(latencies):.1f} ms)...")

            # Salva resultados brutos em JSON
            out_file = os.path.join(output_dir, f"raw_retrievals_{model_id}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_id": model_id,
                        "config": cfg,
                        "total_queries": len(testset),
                        "index_build_time_s": index_build_time,
                        "mean_latency_ms": float(np.mean(latencies)),
                        "records": retrieval_records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[{model_id}] Resultados salvos com sucesso em: {out_file}")

        finally:
            # Liberação explícita de VRAM após cada modelo (OC-5)
            unload_vram(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de Modelos de Embedding")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Modelos a avaliar ('all' ou lista separada por vírgula: openai_3_small,qwen3_8b_q4km,...)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o número de consultas (para testes rápidos / smoke test)",
    )
    args = parser.parse_args()

    if args.models == "all":
        models_to_run = list(MODELS_CONFIG.keys())
    else:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]

    run_benchmark(models_to_run, limit=args.limit)
