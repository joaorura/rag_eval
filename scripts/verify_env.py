"""Script de verificação de sanidade do ambiente para execução no OpenCode."""

import os
import sys
import torch
import requests
from dotenv import load_dotenv

def main():
    print("=== Verificação de Sanidade do Ambiente ===")
    load_dotenv()

    # 1. Checa chave da OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if key and len(key.strip()) > 10:
        print("[OK] OPENAI_API_KEY carregada com sucesso do .env")
    else:
        print("[FALHA] OPENAI_API_KEY ausente ou inválida no .env")

    # 2. Checa CUDA e GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[OK] CUDA disponível: {gpu_name} ({vram_mb:.0f} MiB VRAM)")
    else:
        print("[AVISO] CUDA não detectado pelo PyTorch. Execução será em CPU.")

    # 3. Checa conexão com o Ollama
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            print(f"[OK] Ollama ativo em localhost:11434 com {len(models)} modelo(s):")
            for m in models:
                print(f"     - {m}")
        else:
            print(f"[FALHA] Ollama respondeu com status {res.status_code}")
    except Exception as e:
        print(f"[FALHA] Não foi possível conectar ao Ollama: {e}")

    # 4. Checa corpus e ground truth
    if os.path.exists("data"):
        pdfs = [f for f in os.listdir("data") if f.endswith(".pdf")]
        print(f"[OK] Corpus técnico encontrado em 'data/': {len(pdfs)} arquivos PDF")
    else:
        print("[FALHA] Diretório 'data/' não encontrado!")

    if os.path.exists("testset_openai_4omini.jsonl"):
        print("[OK] Ground truth 'testset_openai_4omini.jsonl' presente")
    else:
        print("[FALHA] 'testset_openai_4omini.jsonl' não encontrado!")

    print("===========================================")

if __name__ == "__main__":
    main()
