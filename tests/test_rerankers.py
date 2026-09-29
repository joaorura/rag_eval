"""Unit tests for Reranker models and utilities."""

import math
from unittest.mock import MagicMock, patch

import pytest
import requests
from llama_index.core.schema import NodeWithScore, TextNode

from scripts.rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    Qwen3OllamaReranker,
    RankGPTReranker,
    compute_binary_softmax,
    parse_ollama_logprob_score,
)


# ============================================================================
# 1. Binary Logprob Softmax Tests
# ============================================================================


class TestBinaryLogprobSoftmax:
    """Tests for binary softmax computation from logprobs."""

    def test_exact_softmax_formula(self):
        """Verify score = exp(lp_yes) / (exp(lp_yes) + exp(lp_no))."""
        lp_yes = -0.5
        lp_no = -1.5
        expected = math.exp(lp_yes) / (math.exp(lp_yes) + math.exp(lp_no))
        actual = compute_binary_softmax(lp_yes, lp_no)
        assert pytest.approx(actual, rel=1e-6) == expected

    def test_equal_logprobs_yields_half(self):
        """Equal logprobs for yes and no should produce 0.5."""
        assert compute_binary_softmax(-2.0, -2.0) == pytest.approx(0.5)
        assert compute_binary_softmax(0.0, 0.0) == pytest.approx(0.5)

    def test_extreme_yes_favored(self):
        """Very high lp_yes relative to lp_no should approach 1.0 without overflow."""
        score = compute_binary_softmax(0.0, -100.0)
        assert score == pytest.approx(1.0, rel=1e-5)

    def test_extreme_no_favored(self):
        """Very high lp_no relative to lp_yes should approach 0.0 without underflow."""
        score = compute_binary_softmax(-100.0, 0.0)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_parse_ollama_response_standard(self):
        """Verify parsing standard Ollama chat response with top_logprobs."""
        mock_response = {
            "message": {"role": "assistant", "content": "yes"},
            "logprobs": [
                {
                    "token": "yes",
                    "logprob": -0.64779,
                    "top_logprobs": [
                        {"token": "yes", "logprob": -0.64779},
                        {"token": "no", "logprob": -4.13395},
                        {"token": "other", "logprob": -5.0},
                    ],
                }
            ],
        }
        expected = math.exp(-0.64779) / (math.exp(-0.64779) + math.exp(-4.13395))
        score = parse_ollama_logprob_score(mock_response)
        assert pytest.approx(score, rel=1e-5) == expected

    def test_parse_ollama_response_case_and_whitespace_insensitivity(self):
        """Verify handling of variations like ' Yes' or 'NO'."""
        mock_response = {
            "message": {"role": "assistant", "content": "yes"},
            "logprobs": [
                {
                    "token": " Yes",
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": " Yes", "logprob": -0.2},
                        {"token": " NO ", "logprob": -2.5},
                    ],
                }
            ],
        }
        expected = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-2.5))
        score = parse_ollama_logprob_score(mock_response)
        assert pytest.approx(score, rel=1e-5) == expected

    def test_parse_ollama_response_only_yes_in_top(self):
        """When 'no' is not in top_logprobs, it should default to a small probability."""
        mock_response = {
            "message": {"role": "assistant", "content": "yes"},
            "logprobs": [
                {
                    "token": "yes",
                    "logprob": -0.05,
                    "top_logprobs": [
                        {"token": "yes", "logprob": -0.05},
                        {"token": "sure", "logprob": -3.5},
                    ],
                }
            ],
        }
        score = parse_ollama_logprob_score(mock_response)
        assert score > 0.99

    def test_parse_ollama_response_only_no_in_top(self):
        """When 'yes' is not in top_logprobs, score should approach 0.0."""
        mock_response = {
            "message": {"role": "assistant", "content": "no"},
            "logprobs": [
                {
                    "token": "no",
                    "logprob": -0.05,
                    "top_logprobs": [
                        {"token": "no", "logprob": -0.05},
                        {"token": "never", "logprob": -4.0},
                    ],
                }
            ],
        }
        score = parse_ollama_logprob_score(mock_response)
        assert score < 0.01

    def test_parse_ollama_response_fallback_to_text(self):
        """When logprobs are missing or empty, fallback to message content."""
        mock_yes = {"message": {"content": "yes"}, "logprobs": []}
        mock_no = {"message": {"content": "no"}, "logprobs": None}
        mock_other = {"message": {"content": "maybe"}}

        assert parse_ollama_logprob_score(mock_yes) == 1.0
        assert parse_ollama_logprob_score(mock_no) == 0.0
        assert parse_ollama_logprob_score(mock_other) == 0.0


# ============================================================================
# 2. Descending Rank Sorting Tests
# ============================================================================


class TestDescendingRankSorting:
    """Tests verifying descending score order and top_n truncation."""

    @pytest.fixture
    def sample_nodes(self):
        return [
            NodeWithScore(node=TextNode(text="Doc 0 - Low relevance"), score=0.1),
            NodeWithScore(node=TextNode(text="Doc 1 - Mid relevance"), score=0.5),
            NodeWithScore(node=TextNode(text="Doc 2 - High relevance"), score=0.9),
        ]

    def test_qwen3_ollama_reranker_sorting(self, sample_nodes):
        """Qwen3OllamaReranker sorts candidate nodes descending by binary softmax score."""
        reranker = Qwen3OllamaReranker()

        # Mock requests.post to return different scores for different documents
        # Doc 0 -> score 0.2
        # Doc 1 -> score 0.8
        # Doc 2 -> score 0.4
        def fake_post(*args, **kwargs):
            doc = kwargs["json"]["messages"][1]["content"]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "Doc 0" in doc:
                mock_resp.json.return_value = {
                    "logprobs": [{"top_logprobs": [{"token": "yes", "logprob": -2.0}, {"token": "no", "logprob": -0.2}]}]
                }
            elif "Doc 1" in doc:
                mock_resp.json.return_value = {
                    "logprobs": [{"top_logprobs": [{"token": "yes", "logprob": -0.1}, {"token": "no", "logprob": -2.5}]}]
                }
            else:
                mock_resp.json.return_value = {
                    "logprobs": [{"top_logprobs": [{"token": "yes", "logprob": -1.0}, {"token": "no", "logprob": -0.8}]}]
                }
            return mock_resp

        with patch("requests.post", side_effect=fake_post):
            results = reranker.rerank(query="potencia nobreak", nodes=sample_nodes, top_n=2)

        assert len(results) == 2
        # Doc 1 has highest score (~0.917), followed by Doc 2 (~0.450)
        assert "Doc 1" in results[0].node.get_content()
        assert "Doc 2" in results[1].node.get_content()
        assert results[0].score > results[1].score

    def test_cross_encoder_reranker_sorting(self, sample_nodes):
        """CrossEncoderReranker sorts candidate nodes descending by model output."""
        mock_model = MagicMock()
        # Predictions for Doc 0, Doc 1, Doc 2
        mock_model.predict.return_value = [0.15, 0.95, 0.45]

        reranker = CrossEncoderReranker(model=mock_model)
        results = reranker.rerank(query="manual instalacao", nodes=sample_nodes, top_n=3)

        assert len(results) == 3
        assert "Doc 1" in results[0].node.get_content()
        assert "Doc 2" in results[1].node.get_content()
        assert "Doc 0" in results[2].node.get_content()
        assert results[0].score == 0.95
        assert results[1].score == 0.45
        assert results[2].score == 0.15

    def test_rankgpt_reranker_sorting(self, sample_nodes):
        """RankGPTReranker sorts nodes descending using postprocessor or LLM."""
        mock_postprocessor = MagicMock()
        reordered = [
            NodeWithScore(node=sample_nodes[2].node, score=0.99),
            NodeWithScore(node=sample_nodes[0].node, score=0.75),
            NodeWithScore(node=sample_nodes[1].node, score=0.30),
        ]
        mock_postprocessor.postprocess_nodes.return_value = reordered

        reranker = RankGPTReranker(postprocessor=mock_postprocessor)
        results = reranker.rerank(query="especificacoes tecnicas", nodes=sample_nodes, top_n=2)

        assert len(results) == 2
        assert "Doc 2" in results[0].node.get_content()
        assert "Doc 0" in results[1].node.get_content()
        assert results[0].score >= results[1].score


# ============================================================================
# 3. Graceful Fallback Tests
# ============================================================================


class TestGracefulFallback:
    """Tests verifying original order and scores are preserved upon failure."""

    @pytest.fixture
    def original_nodes(self):
        return [
            NodeWithScore(node=TextNode(text="First Node"), score=0.9),
            NodeWithScore(node=TextNode(text="Second Node"), score=0.8),
            NodeWithScore(node=TextNode(text="Third Node"), score=0.7),
        ]

    def test_qwen3_ollama_timeout_fallback(self, original_nodes):
        """Qwen3OllamaReranker preserves original node order when requests times out."""
        reranker = Qwen3OllamaReranker()

        with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
            results = reranker.rerank(query="alarme bateria", nodes=original_nodes, top_n=2)

        assert len(results) == 2
        assert results[0].node.get_content() == "First Node"
        assert results[1].node.get_content() == "Second Node"
        assert results[0].score == 0.9
        assert results[1].score == 0.8

    def test_qwen3_ollama_http_error_fallback(self, original_nodes):
        """Qwen3OllamaReranker preserves original node order on HTTP error status."""
        reranker = Qwen3OllamaReranker()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")

        with patch("requests.post", return_value=mock_resp):
            results = reranker.rerank(query="alarme bateria", nodes=original_nodes, top_n=3)

        assert len(results) == 3
        assert [r.node.get_content() for r in results] == ["First Node", "Second Node", "Third Node"]

    def test_cross_encoder_exception_fallback(self, original_nodes):
        """CrossEncoderReranker preserves original node order when inference fails."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA Out of Memory")

        reranker = CrossEncoderReranker(model=mock_model)
        results = reranker.rerank(query="curto circuito", nodes=original_nodes, top_n=2)

        assert len(results) == 2
        assert results[0].node.get_content() == "First Node"
        assert results[1].node.get_content() == "Second Node"

    def test_rankgpt_exception_fallback(self, original_nodes):
        """RankGPTReranker preserves original node order when postprocess fails."""
        mock_postprocessor = MagicMock()
        mock_postprocessor.postprocess_nodes.side_effect = Exception("OpenAI Rate Limit Exceeded")

        reranker = RankGPTReranker(postprocessor=mock_postprocessor)
        results = reranker.rerank(query="manutencao preventiva", nodes=original_nodes, top_n=2)

        assert len(results) == 2
        assert results[0].node.get_content() == "First Node"
        assert results[1].node.get_content() == "Second Node"


# ============================================================================
# 4. Empty Input Handling Tests
# ============================================================================


class TestEmptyInputHandling:
    """Tests verifying behavior with empty lists or empty queries."""

    def test_empty_nodes_qwen3(self):
        """Empty node list returns empty list without calling API."""
        reranker = Qwen3OllamaReranker()
        with patch("requests.post") as mock_post:
            assert reranker.rerank(query="potencia", nodes=[], top_n=5) == []
            mock_post.assert_not_called()

    def test_empty_nodes_cross_encoder(self):
        """Empty node list returns empty list without calling model."""
        mock_model = MagicMock()
        reranker = CrossEncoderReranker(model=mock_model)
        assert reranker.rerank(query="potencia", nodes=[], top_n=5) == []
        mock_model.predict.assert_not_called()

    def test_empty_nodes_rankgpt(self):
        """Empty node list returns empty list without calling postprocessor."""
        mock_postprocessor = MagicMock()
        reranker = RankGPTReranker(postprocessor=mock_postprocessor)
        assert reranker.rerank(query="potencia", nodes=[], top_n=5) == []
        mock_postprocessor.postprocess_nodes.assert_not_called()

    def test_empty_query_string_handled_safely(self):
        """Empty query does not crash and processes candidate nodes."""
        nodes = [NodeWithScore(node=TextNode(text="Passage A"), score=0.5)]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.88]

        reranker = CrossEncoderReranker(model=mock_model)
        res = reranker.rerank(query="", nodes=nodes, top_n=1)
        assert len(res) == 1
        assert res[0].score == 0.88


# ============================================================================
# 5. Ollama Payload Verification Tests
# ============================================================================


class TestOllamaPayloadStructure:
    """Verifies that the Ollama request conforms strictly to the Qwen3 reranker prompt format."""

    def test_payload_fields(self):
        reranker = Qwen3OllamaReranker(
            model_name="dengcao/Qwen3-Reranker-4B:Q4_K_M",
            instruction="Retrieve documents relevant to the query",
        )

        node = NodeWithScore(node=TextNode(text="Conteudo do manual UPS"), score=0.5)

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "logprobs": [{"top_logprobs": [{"token": "yes", "logprob": -0.1}, {"token": "no", "logprob": -2.0}]}]
            }
            mock_post.return_value = mock_resp

            reranker.rerank(query="como ligar", nodes=[node], top_n=1)

            assert mock_post.called
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]

            assert payload["model"] == "dengcao/Qwen3-Reranker-4B:Q4_K_M"
            assert payload["think"] is False
            assert payload["logprobs"] is True
            assert payload["top_logprobs"] == 20
            assert payload["options"]["num_predict"] == 1
            assert payload["options"]["temperature"] == 0

            # Verify system prompt
            sys_msg = payload["messages"][0]
            assert sys_msg["role"] == "system"
            assert 'Note that the answer can only be "yes" or "no".' in sys_msg["content"]

            # Verify user prompt format: <Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}
            user_msg = payload["messages"][1]
            assert user_msg["role"] == "user"
            assert "<Instruct>: Retrieve documents relevant to the query" in user_msg["content"]
            assert "<Query>: como ligar" in user_msg["content"]
            assert "<Document>: Conteudo do manual UPS" in user_msg["content"]


# ============================================================================
# 6. Additional Adapter Methods & Interface Tests
# ============================================================================


class TestAdapterMethodsAndInterface:
    """Verifies BaseReranker interface and additional adapter methods."""

    def test_base_reranker_cannot_be_instantiated(self):
        """BaseReranker is an abstract base class."""
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore

    def test_qwen3_unload(self):
        """Unload method calls Ollama generate endpoint with keep_alive=0."""
        reranker = Qwen3OllamaReranker(model_name="test-model", base_url="http://localhost:11434")
        with patch("requests.post") as mock_post:
            reranker.unload()
            mock_post.assert_called_once_with(
                "http://localhost:11434/api/generate",
                json={"model": "test-model", "keep_alive": 0},
                timeout=10,
            )

    def test_rankgpt_listwise_fallback(self):
        """RankGPTReranker executes listwise fallback correctly when postprocessor is None."""
        mock_llm = MagicMock()
        mock_completion = MagicMock()
        mock_completion.text = "[2, 1]"
        mock_llm.complete.return_value = mock_completion

        nodes = [
            NodeWithScore(node=TextNode(text="Passage 1"), score=0.4),
            NodeWithScore(node=TextNode(text="Passage 2"), score=0.5),
        ]

        reranker = RankGPTReranker(llm=mock_llm)
        # Force postprocessor to None to test fallback
        with patch.object(RankGPTReranker, "postprocessor", new=None):
            result = reranker.rerank(query="consulta", nodes=nodes, top_n=2)

        assert len(result) == 2
        assert result[0].node.get_content() == "Passage 2"
        assert result[1].node.get_content() == "Passage 1"

