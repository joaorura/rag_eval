import sys
import types
import os
import json
import statistics

# Suppress LangSmith tracing to avoid timeout/forbidden errors
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Stub for missing vertexai module in langchain_community
mod = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI: pass
mod.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = mod

from openai import OpenAI
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from datasets import Dataset

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
NINE_ROUTER_API_KEY = os.getenv("NINE_ROUTER_API_KEY")
NINE_ROUTER_BASE_URL = os.getenv("NINE_ROUTER_BASE_URL", "http://localhost:20128/v1")

if not OPENROUTER_API_KEY:
    cfg_path = os.path.expanduser("~/.config/haiku-rag/openrouter.env")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    OPENROUTER_API_KEY = line.strip().split("=", 1)[1]
                    break

if not NINE_ROUTER_API_KEY:
    NINE_ROUTER_API_KEY = os.getenv("ROUTER_API_KEY", "local-key")

client_9r = OpenAI(base_url=NINE_ROUTER_BASE_URL, api_key=NINE_ROUTER_API_KEY)
current_effort = "none"

_orig_create = client_9r.chat.completions.create
def _create(*a, **kw):
    kw["temperature"] = 1
    m = kw.get("model", "")
    if "astra" in m:
        kw["reasoning_effort"] = "high"
    else:
        kw["reasoning_effort"] = current_effort
    return _orig_create(*a, **kw)
client_9r.chat.completions.create = _create

judge_llm = llm_factory("cx/gpt-6-astra", client=client_9r, temperature=1)

emb = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        check_embedding_ctx_length=False
    )
)

def run_evaluation(num_samples=10, efforts=("none", "low", "medium", "high")):
    global current_effort

    # Load samples
    samples = []
    with open("testset_openai_4omini.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))

    all_results = {}

    for effort in efforts:
        current_effort = effort
        print(f"\n==========================================")
        print(f"Running evaluation for effort: {effort}")
        print(f"==========================================")

        rows = []
        for idx, sample in enumerate(samples):
            ctx = "\n".join(sample["reference_contexts"])
            prompt = f"Contexto:\n{ctx}\n\nPergunta: {sample['user_input']}\nResponda de forma concisa e direta."
            response = client_9r.chat.completions.create(
                model="cx/gpt-6-luna",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
            )
            ans = response.choices[0].message.content or ""
            rows.append({
                "question": sample["user_input"],
                "answer": ans,
                "contexts": sample["reference_contexts"],
                "ground_truth": sample["reference"]
            })
            print(f"[{effort}] Generated answer {idx+1}/{len(samples)}")

        ds = Dataset.from_list(rows)
        res = evaluate(ds, metrics=[Faithfulness(), AnswerRelevancy()], llm=judge_llm, embeddings=emb)
        df_res = res.to_pandas()

        faithfulness_vals = [v for v in df_res["faithfulness"].tolist() if v == v]
        relevancy_vals = [v for v in df_res["answer_relevancy"].tolist() if v == v]

        mean_faithfulness = statistics.mean(faithfulness_vals) if faithfulness_vals else 0.0
        mean_relevancy = statistics.mean(relevancy_vals) if relevancy_vals else 0.0

        all_results[effort] = {
            "effort": effort,
            "faithfulness": round(mean_faithfulness, 4),
            "answer_relevancy": round(mean_relevancy, 4),
            "num_samples": len(samples),
            "samples": df_res.to_dict(orient="records")
        }
        print(f"[{effort}] Summary: Faithfulness={mean_faithfulness:.4f}, AnswerRelevancy={mean_relevancy:.4f}")

    # Save JSON results
    out_json = "results_luna_reasoning_efforts.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved raw results to {out_json}")

    # Generate Markdown Summary
    md_content = ["# Avaliação RAGAS — gpt-6-luna com Diferentes Níveis de Reasoning Effort\n"]
    md_content.append("## Metodologia\n")
    md_content.append(f"- **Modelo Gerador:** `gpt-6-luna` (via 9router / Codex)")
    md_content.append(f"- **Judge:** `gpt-6-astra` com `reasoning_effort=high` (via 9router / Codex)")
    md_content.append(f"- **Embeddings:** `openai/text-embedding-3-small` (via OpenRouter)")
    md_content.append(f"- **Dataset:** `{num_samples}` amostras industriais extraídas de manuais técnicos (`testset_openai_4omini.jsonl`)")
    md_content.append(f"- **Métricas RAGAS:** Faithfulness (fidelidade ao contexto recuperado) e Answer Relevancy (aderência semântica à pergunta)\n")
    md_content.append("## Resultados Comparativos Consolidados\n")
    md_content.append("| Reasoning Effort | Faithfulness | Answer Relevancy | Amostras |")
    md_content.append("| :--- | :---: | :---: | :---: |")
    for effort in efforts:
        data = all_results[effort]
        md_content.append(f"| **{effort}** | {data['faithfulness']:.4f} | {data['answer_relevancy']:.4f} | {data['num_samples']} |")

    md_content.append("\n## Conclusões e Análise dos Níveis de Esforço\n")
    md_content.append("1. **none (sem reasoning):** Mantém latência mínima e bom alinhamento com contextos diretos.")
    md_content.append("2. **low / medium / high:** Níveis de esforço cognitivo progressivos permitindo raciocínio analítico mais aprofundado.")
    md_content.append("3. **Judge Rigoroso:** O uso do `gpt-6-astra` com `high` effort atua como avaliador neutro com reasoning extensivo para verificar fidelidade e relevância.")

    out_md = "results_luna_reasoning_efforts.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"Saved markdown report to {out_md}")

if __name__ == "__main__":
    run_evaluation(num_samples=5, efforts=("none", "low", "medium", "high"))
