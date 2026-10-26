#!/usr/bin/env bash
set -e

cd /home/joaorura/orca/workspaces/rag_eval/gorgonian
echo "[$(date)] Starting evaluation of gpt-6-sol across reasoning efforts..."

/home/joaorura/miniconda3/bin/python scripts/evaluate_sol_efforts.py

echo "[$(date)] Evaluation finished. Committing and pushing results to joaorura/validacao-luna..."
git add -f results_sol_reasoning_efforts.json results_sol_reasoning_efforts.md scripts/evaluate_sol_efforts.py
git commit -m "feat(benchmark): add RAGAS evaluation results for gpt-6-sol across reasoning efforts" || true
git push origin joaorura/validacao-luna

echo "[$(date)] Sol benchmark done!"
