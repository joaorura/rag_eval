"""Testes unitários e de integração para scripts/qualitative_audit.py."""

import json
import os
import tempfile
import pytest

from scripts.qualitative_audit import (
    truncate_text,
    extract_reference_text,
    load_testset,
    parse_existing_classifications,
    run_audit,
    CLASSIFICATION_PENDING,
)


def test_truncate_text():
    # Texto menor que limite
    assert truncate_text("Texto curto", 20) == "Texto curto"
    # Texto com pipes e quebras de linha
    assert truncate_text("Linha 1\nLinha 2 | Linha 3", 50) == "Linha 1 Linha 2 / Linha 3"
    # Texto maior que limite
    long_text = "a" * 85
    truncated = truncate_text(long_text, 80)
    assert len(truncated) == 80
    assert truncated.endswith("...")
    assert truncated == ("a" * 77) + "..."
    # Texto vazio / None
    assert truncate_text("", 50) == ""
    assert truncate_text(None, 50) == ""


def test_extract_reference_text():
    # Com contexts
    sample_with_contexts = {
        "reference_contexts": ["Trecho 1 com detalhes", "Trecho 2 com especificações"],
        "reference": "Resposta sintética",
    }
    extracted = extract_reference_text(sample_with_contexts)
    assert extracted == "Trecho 1 com detalhes / Trecho 2 com especificações"

    # Sem contexts (fallback para reference)
    sample_fallback = {
        "reference_contexts": [],
        "reference": "Resposta sintética esperada",
    }
    assert extract_reference_text(sample_fallback) == "Resposta sintética esperada"


def test_audit_deterministic_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "auditoria_test.md")
        testset_file = os.path.join(tmpdir, "testset.jsonl")

        # Cria 30 amostras sintéticas
        with open(testset_file, "w", encoding="utf-8") as f:
            for i in range(30):
                f.write(
                    json.dumps(
                        {
                            "user_input": f"Pergunta de teste número {i:02d} com texto suficiente para avaliar",
                            "reference_contexts": [f"Contexto do documento número {i:02d}"],
                            "reference": f"Resposta número {i:02d}",
                            "synthesizer_name": "TestSynthesizer",
                        }
                    )
                    + "\n"
                )

        # Executa auditoria com seed=42 e 20 amostras
        run_audit(
            testset_path=testset_file,
            output_path=output_file,
            seed=42,
            num_samples=20,
            overwrite=True,
        )

        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verifica cabeçalhos e estrutura essencial
        assert "# Auditoria Qualitativa do Ground Truth (20 Amostras)" in content
        assert "## 1. Tabela de Amostras para Avaliação" in content
        assert "## 2. Estatísticas do Resumo" in content
        assert "## 3. Conclusão sobre o Nível de Viés Circular" in content
        assert "## 4. Detalhamento Integral das Amostras para Auditoria" in content
        assert "Sample #" in content
        assert "Classificação" in content
        assert CLASSIFICATION_PENDING in content

        # Simula preenchimento manual de avaliações pelo usuário
        modified_content = content.replace(CLASSIFICATION_PENDING, "Relevante", 1)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(modified_content)

        # Executa novamente sem overwrite: deve preservar 'Relevante' e calcular estatísticas
        run_audit(
            testset_path=testset_file,
            output_path=output_file,
            seed=42,
            num_samples=20,
            overwrite=False,
        )

        with open(output_file, "r", encoding="utf-8") as f:
            updated_content = f.read()

        assert "| Relevante | 1 / 20 | 5.0% |" in updated_content
