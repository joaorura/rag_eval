#!/usr/bin/env python3
"""Auditoria Qualitativa de Amostras do Ground Truth para Mitigação de Viés Circular.

Este script realiza a amostragem aleatória reproduzível (seed=42) de 20 instâncias
do conjunto de teste sintético (testset_openai_4omini.jsonl) e gera um relatório
estruturado em Markdown em results/auditoria_qualitativa_20amostras.md.

Finalidade Metodológica (TCC):
Mitigar e analisar o viés circular (circular bias) decorrente da geração do ground
truth utilizando GPT-4o-mini + OpenAI embeddings, que pode favorecer a família OpenAI.
O relatório apresenta uma tabela formatada com as colunas prontas para avaliação manual:
Sample #, Query (truncada a 80c), Reference (truncada a 100c) e Classificação.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
from typing import Any

# Adiciona o diretório raiz ao sys.path para importações relativas
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

DEFAULT_TESTSET_PATH = os.path.join(PROJECT_ROOT, "testset_openai_4omini.jsonl")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "auditoria_qualitativa_20amostras.md")
DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 20

CLASSIFICATION_PENDING = "⬜ A avaliar"
VALID_CLASSIFICATIONS = ["Relevante", "Parcialmente Relevante", "Irrelevante"]


def truncate_text(text: str, max_chars: int) -> str:
    """Normaliza espaços em branco e trunca o texto adicionando '...' se exceder max_chars."""
    if not text:
        return ""
    # Substitui pipes para evitar quebras em tabelas Markdown
    cleaned = " ".join(str(text).replace("|", "/").split())
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 3] + "..."
    return cleaned


def extract_reference_text(sample: dict[str, Any]) -> str:
    """Extrai texto representativo dos contextos de referência esperados (reference_contexts)."""
    contexts = sample.get("reference_contexts", [])
    if contexts and isinstance(contexts, list):
        cleaned_ctxs = [" ".join(c.split()) for c in contexts if isinstance(c, str) and c.strip()]
        if cleaned_ctxs:
            return " / ".join(cleaned_ctxs)
    # Fallback para o campo reference sintetizado caso reference_contexts não exista
    return str(sample.get("reference", "")).strip()


def load_testset(filepath: str) -> list[dict[str, Any]]:
    """Carrega o testset em formato JSONL a partir do caminho fornecido."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo de testset não encontrado: {filepath}")

    samples: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                data["_line_number"] = line_num
                data["_index"] = line_num - 1
                samples.append(data)
            except json.JSONDecodeError as e:
                print(f"Aviso: Linha {line_num} inválida no JSONL: {e}", file=sys.stderr)

    return samples


def parse_existing_classifications(output_file: str) -> dict[int, str]:
    """Extrai classificações manuais previamente preenchidas no relatório Markdown existente."""
    if not os.path.exists(output_file):
        return {}

    classifications: dict[int, str] = {}
    pattern = re.compile(r"\|\s*(\d+)\s*\|.*?\|.*?\|\s*([^|]+?)\s*\|")

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                sample_num = int(match.group(1))
                val = match.group(2).strip()
                # Se não for o placeholder inicial ou cabeçalho da tabela
                if val and val != "Classificação" and not val.startswith("---"):
                    classifications[sample_num] = val

    return classifications


def generate_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Gera tabela Markdown a partir das linhas fornecidas, usando pandas ou fallback nativo."""
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        return df.to_markdown(index=False)

    # Fallback sem pandas
    headers = ["Sample #", "Query", "Reference", "Classificação"]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    lines = [
        "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |",
        "| " + " | ".join("-" * col_widths[h] for h in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
            + " |"
        )
    return "\n".join(lines)


def build_report_content(
    sampled_items: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    total_testset_size: int,
    seed: int,
) -> str:
    """Constrói o relatório estruturado completo em formato Markdown."""
    generation_date = datetime.date.today().strftime("%Y-%m-%d")
    md_table = generate_markdown_table(table_rows)

    # Contagem de status das classificações
    counts = {cat: 0 for cat in VALID_CLASSIFICATIONS}
    counts[CLASSIFICATION_PENDING] = 0
    outros = 0

    for r in table_rows:
        val = r["Classificação"].strip()
        matched = False
        for cat in VALID_CLASSIFICATIONS:
            if cat.lower() in val.lower():
                counts[cat] += 1
                matched = True
                break
        if not matched:
            if CLASSIFICATION_PENDING in val or "avaliar" in val.lower():
                counts[CLASSIFICATION_PENDING] += 1
            else:
                outros += 1

    total_avaliados = sum(counts[cat] for cat in VALID_CLASSIFICATIONS)
    is_fully_pending = (total_avaliados == 0)

    # Seção de Estatísticas do Resumo
    if is_fully_pending:
        stats_section = (
            "| Classificação | Quantidade (N) | Proporção (%) |\n"
            "|:---|:---:|:---:|\n"
            "| Relevante | `__` / 20 | `__%` |\n"
            "| Parcialmente Relevante | `__` / 20 | `__%` |\n"
            "| Irrelevante | `__` / 20 | `__%` |\n"
            "| **Total Avaliado** | **`0` / 20** | **`0.0%`** |\n\n"
            f"> *Nota: As 20 amostras estão atualmente com status `{CLASSIFICATION_PENDING}`. "
            "Preencha a coluna 'Classificação' na tabela e execute novamente o script "
            "para calcular automaticamente as estatísticas consolidadas.*"
        )
    else:
        n_tot = len(table_rows)
        pct_rel = (counts["Relevante"] / n_tot) * 100
        pct_parc = (counts["Parcialmente Relevante"] / n_tot) * 100
        pct_irrel = (counts["Irrelevante"] / n_tot) * 100
        pct_avaliados = (total_avaliados / n_tot) * 100

        stats_section = (
            "| Classificação | Quantidade (N) | Proporção (%) |\n"
            "|:---|:---:|:---:|\n"
            f"| Relevante | {counts['Relevante']} / {n_tot} | {pct_rel:.1f}% |\n"
            f"| Parcialmente Relevante | {counts['Parcialmente Relevante']} / {n_tot} | {pct_parc:.1f}% |\n"
            f"| Irrelevante | {counts['Irrelevante']} / {n_tot} | {pct_irrel:.1f}% |\n"
            f"| **Total Avaliado** | **{total_avaliados} / {n_tot}** | **{pct_avaliados:.1f}%** |\n\n"
            f"> *Amostras pendentes de avaliação: {counts[CLASSIFICATION_PENDING] + outros} / {n_tot}.*"
        )

    # Detalhamento de cada amostra para permitir a inspeção do contexto completo
    sample_details: list[str] = []
    for i, item in enumerate(sampled_items, start=1):
        idx = item.get("_index", i - 1)
        line_no = item.get("_line_number", idx + 1)
        query = item.get("user_input", "").strip()
        ref_ans = item.get("reference", "").strip()
        synth = item.get("synthesizer_name", "N/A")
        raw_contexts = item.get("reference_contexts", [])

        # Formatação dos contextos
        ctx_blocks = []
        if isinstance(raw_contexts, list) and raw_contexts:
            if len(raw_contexts) <= 3:
                for c_idx, c in enumerate(raw_contexts, start=1):
                    clean_c = c.strip()
                    ctx_blocks.append(f"  **Contexto {c_idx}:**\n  ```text\n  {clean_c}\n  ```")
            else:
                for c_idx, c in enumerate(raw_contexts[:3], start=1):
                    clean_c = c.strip()
                    ctx_blocks.append(f"  **Contexto {c_idx}:**\n  ```text\n  {clean_c}\n  ```")
                remaining_ctxs = []
                for c_idx, c in enumerate(raw_contexts[3:], start=4):
                    clean_c = c.strip()
                    remaining_ctxs.append(f"  **Contexto {c_idx}:**\n  ```text\n  {clean_c}\n  ```")
                extra_text = "\n".join(remaining_ctxs)
                ctx_blocks.append(
                    f"  <details>\n  <summary><i>Ver mais {len(raw_contexts) - 3} contextos adicionais...</i></summary>\n\n"
                    f"{extra_text}\n  </details>"
                )
        else:
            ctx_blocks.append("  *(Nenhum contexto de referência encontrado no item)*")

        contexts_rendered = "\n\n".join(ctx_blocks)
        curr_class = table_rows[i - 1]["Classificação"]

        detail_md = (
            f"### Amostra {i} (Índice no testset: `{idx}` | Linha: `{line_no}`)\n\n"
            f"- **Pergunta (`user_input`)**:\n  > {query}\n"
            f"- **Tipo de Sintetizador**: `{synth}`\n"
            f"- **Contextos de Referência Esperados (`reference_contexts` - {len(raw_contexts)} trecho(s))**:\n\n"
            f"{contexts_rendered}\n\n"
            f"- **Resposta de Referência Sintetizada (`reference`)**:\n  > {ref_ans}\n"
            f"- **Classificação**: `{curr_class}`\n"
            f"- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`\n"
        )
        sample_details.append(detail_md)

    details_joined = "\n---\n\n".join(sample_details)

    report = f"""# Auditoria Qualitativa do Ground Truth (20 Amostras)

## Metadados e Contexto Metodológico
- **Data de Execução**: {generation_date}
- **Arquivo Fonte**: `testset_openai_4omini.jsonl` (Total: {total_testset_size} amostras)
- **Tamanho da Amostra**: {len(table_rows)} amostras aleatórias (`seed={seed}`)
- **Finalidade Científica**: Mitigação e quantificação do **viés circular (*circular bias*)** no benchmark de recuperação vetorial para o TCC.

### Por que a auditoria qualitativa manual é necessária?
O conjunto de avaliação sintético foi sintetizado com o pipeline Ragas utilizando o gerador **GPT-4o-mini** e indexador com **OpenAI embeddings** (`text-embedding-3-small`). Esse arranjo metodológico pode introduzir um favorecimento sistemático (viés circular) para modelos da mesma família em detrimento de modelos open-source (como LLaMA, DeepSeek e Mixtral).

A auditoria humana manual substitui o uso de juiz LLM adicional (que introduziria um segundo viés) e avalia diretamente:
1. Se a pergunta (`user_input`) possui formulação técnica coerente com o domínio de No-Breaks / sistemas de energia.
2. Se as passagens extraídas (`reference_contexts`) contêm de fato a resposta substantiva esperada.
3. Se a associação decorre de artefatos de chunking (e.g. cabeçalhos vazios, logotipos, fragmentos incompletos) ou de alucinações do modelo gerador.

### Critérios Operacionais de Classificação:
- **`Relevante`**: O contexto de referência contém a resposta direta e substantiva para a pergunta formulada.
- **`Parcialmente Relevante`**: O contexto aborda o tópico geral da consulta, porém omite informações essenciais ou apresenta conteúdo vago/incompleto.
- **`Irrelevante`**: O contexto não responde à pergunta; resulta de corte indevido de texto (ex: cabeçalhos isolados como *"CM Comandos Lineares"*) ou a pergunta deriva de inferência alucinada do LLM.

---

## 1. Tabela de Amostras para Avaliação

{md_table}

---

## 2. Estatísticas do Resumo

{stats_section}

---

## 3. Conclusão sobre o Nível de Viés Circular

### Critérios de Diagnóstico de Viés para o TCC:
- **Validade Alta / Baixo Viés (>80% Relevante)**:
  O conjunto sintético possui representatividade substantiva do corpus técnico. A superioridade de modelos de ponta decorre primordialmente da qualidade do espaço latente e não de artefatos metodológicos.
- **Validade Média / Viés Moderado (60% a 80% Relevante)**:
  Verifica-se incidência relevante de perguntas com contextos parciais ou ruídos de extração. Métricas absolutas (MRR, Hit Rate) podem estar infladas para o modelo baseline da OpenAI, tornando indispensável o uso de testes não-paramétricos (Wilcoxon pareado) para validação estatística.
- **Validade Baixa / Alto Viés (<60% Relevante)**:
  Contaminação substancial do conjunto de testes. A aderência dos modelos deve ser interpretada com forte ressalva metodológica na discussão de resultados da monografia.

### Síntese da Avaliação Manual (Template para a Monografia):
- **Diagnóstico Final de Validade**: `[ ] Alta (>80% Relevante) | [ ] Média (60-80% Relevante) | [ ] Baixa (<60% Relevante)`
- **Padrões de Falha e Artefatos Observados**:
  - *Ruídos de Chunking*: [e.g., cabeçalho institucional 'CM Comandos Lineares' sintetizado incorretamente como comando de programação/automação]
  - *Contextos Genéricos*: [e.g., trechos puramente comerciais ou notas de rodapé de catálogo sem informação técnica]
- **Impacto no Ranking Comparativo dos 7 Modelos de Embedding**:
  - [Avaliar em que medida os modelos open-source foram penalizados por artefatos que coincidem com os padrões gerados pela API da OpenAI]
- **Recomendação para a Redação do TCC**:
  - [Registrar explicitamente esta auditoria no capítulo de Metodologia e Limitações do Trabalho]

---

## 4. Detalhamento Integral das Amostras para Auditoria

{details_joined}
"""
    return report.strip() + "\n"


def run_audit(
    testset_path: str = DEFAULT_TESTSET_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = DEFAULT_SEED,
    num_samples: int = DEFAULT_SAMPLE_SIZE,
    overwrite: bool = False,
) -> None:
    """Executa a rotina de amostragem e geração/atualização do relatório de auditoria."""
    print("=" * 70)
    print("AUDITORIA QUALITATIVA DE AMOSTRAS DO GROUND TRUTH (TCC)")
    print("=" * 70)

    # 1. Carrega o conjunto de testes
    print(f"[*] Carregando testset de: {testset_path}")
    all_samples = load_testset(testset_path)
    total_count = len(all_samples)
    print(f"[+] Total de instâncias no testset: {total_count}")

    if total_count < num_samples:
        raise ValueError(
            f"O testset possui apenas {total_count} amostras, menor que a quantidade requisitada ({num_samples})."
        )

    # 2. Amostragem aleatória com seed fixo (seed=42)
    print(f"[*] Selecionando {num_samples} amostras aleatórias (seed={seed})...")
    random.seed(seed)
    sampled_indices = random.sample(range(total_count), num_samples)
    sampled_items = [all_samples[idx] for idx in sampled_indices]
    print(f"[+] Índices selecionados: {sampled_indices}")

    # 3. Verifica existência de classificações prévias se overwrite não foi forçado
    existing_classifications: dict[int, str] = {}
    if os.path.exists(output_path) and not overwrite:
        print(f"[*] Verificando classificações existentes em: {output_path}")
        existing_classifications = parse_existing_classifications(output_path)
        if existing_classifications:
            evaluated_count = sum(
                1 for v in existing_classifications.values()
                if any(cat.lower() in v.lower() for cat in VALID_CLASSIFICATIONS)
            )
            print(f"[+] Encontradas {len(existing_classifications)} classificações prévias ({evaluated_count} avaliadas).")

    # 4. Prepara linhas da tabela
    table_rows: list[dict[str, Any]] = []
    for i, item in enumerate(sampled_items, start=1):
        query_text = truncate_text(item.get("user_input", ""), 80)
        ref_text = truncate_text(extract_reference_text(item), 100)

        # Se houver classificação prévia mantida, usa-a; caso contrário, define como pendente
        classification = existing_classifications.get(i, CLASSIFICATION_PENDING)

        table_rows.append(
            {
                "Sample #": i,
                "Query": query_text,
                "Reference": ref_text,
                "Classificação": classification,
            }
        )

    # 5. Constrói o relatório em Markdown
    report_content = build_report_content(
        sampled_items=sampled_items,
        table_rows=table_rows,
        total_testset_size=total_count,
        seed=seed,
    )

    # Garante que o diretório de destino exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[+] Relatório de auditoria salvo com sucesso em: {output_path}")

    # 6. Exibe a tabela no terminal para conferência rápida
    print("\n" + "=" * 70)
    print("TABELA DE AUDITORIA GERADA (20 Amostras):")
    print("=" * 70)
    print(generate_markdown_table(table_rows))
    print("=" * 70 + "\n")


def main() -> None:
    """Ponto de entrada CLI."""
    parser = argparse.ArgumentParser(
        description="Auditoria qualitativa de amostras do ground truth (mitigação de viés circular)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_TESTSET_PATH,
        help="Caminho para o arquivo JSONL de teste (padrão: testset_openai_4omini.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Caminho para o relatório Markdown de saída (padrão: results/auditoria_qualitativa_20amostras.md).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semente do gerador aleatório para reprodutibilidade (padrão: 42).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Número de amostras a selecionar (padrão: 20).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Força sobrescrita e reinicialização das classificações para '⬜ A avaliar'.",
    )

    args = parser.parse_args()
    run_audit(
        testset_path=args.input,
        output_path=args.output,
        seed=args.seed,
        num_samples=args.num_samples,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
