import unittest
from scripts.matching_engine import is_context_match, evaluate_query_retrieval

class TestMatchingEngine(unittest.TestCase):
    def test_substring_containment(self):
        ref = "Os No-Breaks da CM Comandos são indicados para aplicações críticas."
        node = "Texto antes. Os No-Breaks da CM Comandos são indicados para aplicações críticas. Texto depois."
        matched, method = is_context_match(node, ref)
        self.assertTrue(matched)
        self.assertEqual(method, "substring")

    def test_fuzzy_matching(self):
        ref = "Equipamentos com tecnologia DSP de alto desempenho para missão crítica."
        node = "Equipamentos com tecnologia DSP de altíssimo desempenho para missao critica."
        matched, method = is_context_match(node, ref)
        self.assertTrue(matched)
        self.assertTrue(method.startswith("partial_") or method.startswith("token_set_"))

    def test_rouge_matching(self):
        ref = "Alta confiabilidade e produtividade nas mais variadas aplicações industriais."
        node = "Garante alta confiabilidade e grande produtividade nas mais variadas aplicações industriais da empresa."
        matched, method = is_context_match(node, ref)
        self.assertTrue(matched)

    def test_negative_matching(self):
        ref = "Especificações elétricas do transformador trifásico isolador."
        node = "Manual de instalação do ar condicionado de janela da sala de reuniões."
        matched, method = is_context_match(node, ref)
        self.assertFalse(matched)
        self.assertEqual(method, "none")

    def test_metrics_calculation(self):
        ref_contexts = [
            "Os nobreaks da linha Nobre possuem modulação PWM de alta frequência.",
            "As baterias seladas VRLA requerem recarga periódica a cada três meses."
        ]
        retrieved_nodes = [
            {"text": "Instalação mecânica recomendada em piso nivelado e ambiente refrigerado."},
            {"text": "Equipamento dotado de tecnologia moderna. Os nobreaks da linha Nobre possuem modulação PWM de alta frequência em sua etapa inversora."},
            {"text": "Relatório de conformidade ambiental de resíduos plásticos."},
            {"text": "Manutenção preventiva das baterias seladas VRLA requerem recarga periódica a cada três meses com corrente controlada."},
            {"text": "Final do manual de instruções do usuário."}
        ]
        metrics = evaluate_query_retrieval(retrieved_nodes, ref_contexts, k_values=[2, 5])
        
        # No Top-2: 
        # Rank 1: irrelevante
        # Rank 2: casa com ref 1 (hit=1.0, recall=0.5, mrr=1/2=0.5)
        self.assertEqual(metrics["hit_rate@2"], 1.0)
        self.assertEqual(metrics["recall@2"], 0.5)
        self.assertEqual(metrics["mrr@2"], 0.5)

        # No Top-5:
        # Rank 4: casa com ref 2 (hit=1.0, recall=1.0, mrr=0.5)
        self.assertEqual(metrics["hit_rate@5"], 1.0)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["mrr@5"], 0.5)

if __name__ == "__main__":
    unittest.main()
