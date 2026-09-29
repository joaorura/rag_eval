import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scripts.matching_engine import evaluate_query_retrieval, load_and_deduplicate_testset

# 1. Load testset
testset_path = "testset_openai_4omini.jsonl"
samples, total_raw, total_dedup = load_and_deduplicate_testset(testset_path)
print(f"Loaded {len(samples)} deduplicated test queries (from {total_raw} raw).")

# 2. Load corpus nodes/chunks
docstore_path = "storage/index_multilingual_e5_large/docstore.json"
with open(docstore_path, "r", encoding="utf-8") as f:
    docstore = json.load(f)

nodes_dict = docstore.get("docstore/data", {})
corpus_node_ids = []
corpus_texts = []
for nid, ndata in nodes_dict.items():
    text = ndata.get("__data__", {}).get("text", "")
    if text:
        corpus_node_ids.append(nid)
        corpus_texts.append(f"passage: {text}")

print(f"Loaded {len(corpus_texts)} corpus chunks from docstore.")

models_to_test = [
    {
        "model_id": "multilingual_e5_large_onnx_int8",
        "repo": "intfloat/multilingual-e5-large",
        "file_name": "model_qint8_avx512_vnni.onnx",
        "dim": 1024,
        "quant": "INT8 (ONNX AVX-512 VNNI)"
    },
    {
        "model_id": "multilingual_e5_base_onnx_int8",
        "repo": "intfloat/multilingual-e5-base",
        "file_name": "model_qint8_avx512_vnni.onnx",
        "dim": 768,
        "quant": "INT8 (ONNX AVX-512 VNNI)"
    }
]

results_summary = []

for m_cfg in models_to_test:
    print(f"\n=======================================================")
    print(f"Benchmarking: {m_cfg['model_id']} ({m_cfg['quant']})")
    print(f"=======================================================")
    
    t0_load = time.perf_counter()
    model = SentenceTransformer(
        m_cfg["repo"],
        backend="onnx",
        model_kwargs={"file_name": m_cfg["file_name"]}
    )
    load_time = time.perf_counter() - t0_load
    print(f"Model loaded in {load_time:.2f} s")
    
    # Encode corpus
    t0_idx = time.perf_counter()
    corpus_embeddings = model.encode(corpus_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    idx_time = time.perf_counter() - t0_idx
    print(f"Corpus ({len(corpus_texts)} docs) encoded in {idx_time:.2f} s")
    
    # Evaluate queries
    query_latencies = []
    records = []
    
    for sample in samples:
        query_text = f"query: {sample['user_input']}"
        t0_q = time.perf_counter()
        q_emb = model.encode([query_text], normalize_embeddings=True)[0]
        
        # Cosine similarity
        scores = np.dot(corpus_embeddings, q_emb)
        top_k_indices = np.argsort(scores)[::-1][:10]
        lat_ms = (time.perf_counter() - t0_q) * 1000
        query_latencies.append(lat_ms)
        
        retrieved_nodes = []
        for rank, idx in enumerate(top_k_indices):
            raw_text = corpus_texts[idx].replace("passage: ", "", 1)
            retrieved_nodes.append({
                "node_id": corpus_node_ids[idx],
                "text": raw_text,
                "score": float(scores[idx]),
                "rank": rank + 1
            })
            
        metrics = evaluate_query_retrieval(retrieved_nodes, sample["reference_contexts"], k_values=[2, 5, 10])
        records.append({
            "query": sample["user_input"],
            "latency_ms": lat_ms,
            "metrics": metrics,
            "retrieved_nodes": retrieved_nodes
        })
        
    mean_lat = float(np.mean(query_latencies))
    hr5 = float(np.mean([r["metrics"]["hit_rate@5"] for r in records]))
    mrr5 = float(np.mean([r["metrics"]["mrr@5"] for r in records]))
    rec5 = float(np.mean([r["metrics"]["recall@5"] for r in records]))
    map5 = float(np.mean([r["metrics"]["map@5"] for r in records]))
    
    hr2 = float(np.mean([r["metrics"]["hit_rate@2"] for r in records]))
    mrr2 = float(np.mean([r["metrics"]["mrr@2"] for r in records]))
    hr10 = float(np.mean([r["metrics"]["hit_rate@10"] for r in records]))
    mrr10 = float(np.mean([r["metrics"]["mrr@10"] for r in records]))
    
    res = {
        "model_id": m_cfg["model_id"],
        "quantization": m_cfg["quant"],
        "dim": m_cfg["dim"],
        "mean_latency_ms": mean_lat,
        "index_build_time_s": idx_time,
        "hr@2": hr2, "mrr@2": mrr2,
        "hr@5": hr5, "mrr@5": mrr5, "rec@5": rec5, "map@5": map5,
        "hr@10": hr10, "mrr@10": mrr10
    }
    results_summary.append(res)
    
    out_json = f"results/raw_retrievals_{m_cfg['model_id']}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model_id": m_cfg["model_id"], "records": records}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")
    print(f"Result (K=5): HR@5={hr5:.4f}, MRR@5={mrr5:.4f}, Recall@5={rec5:.4f}, MAP@5={map5:.4f}, Latency={mean_lat:.2f} ms")

print("\n\n================ CONSOLIDATED ONNX RESULTS ================")
import pandas as pd
df_res = pd.DataFrame(results_summary)
print(df_res[["model_id", "quantization", "dim", "hr@5", "mrr@5", "rec@5", "map@5", "mean_latency_ms"]].to_markdown(index=False))
df_res.to_csv("results/onnx_embeddings_metrics.csv", index=False)
print("Saved results/onnx_embeddings_metrics.csv")
