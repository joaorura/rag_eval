"""Reranker models and utilities for two-stage RAG evaluation."""

from abc import ABC, abstractmethod
import logging
import math
import os
import re
from typing import Any

import requests
import torch
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


def compute_binary_softmax(lp_yes: float, lp_no: float) -> float:
    """Computes binary softmax score for 'yes' vs 'no' logprobs.

    Formula: score = exp(lp_yes) / (exp(lp_yes) + exp(lp_no))
    Evaluated as 1 / (1 + exp(lp_no - lp_yes)) for numerical stability.

    Args:
        lp_yes: Log-probability of token 'yes'.
        lp_no: Log-probability of token 'no'.

    Returns:
        Calibrated binary probability between 0.0 and 1.0.
    """
    diff = lp_no - lp_yes
    if diff > 100.0:
        return 0.0
    if diff < -100.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(diff))


def parse_ollama_logprob_score(response_data: dict[str, Any]) -> float:
    """Parses binary 'yes' vs 'no' logprobs from an Ollama chat response.

    Args:
        response_data: JSON response dict returned by Ollama /api/chat.

    Returns:
        Binary softmax score representing relevance.
    """
    logprobs_list = response_data.get("logprobs")
    if not logprobs_list or not isinstance(logprobs_list, list):
        content = (
            response_data.get("message", {}).get("content", "").strip().lower()
        )
        return 1.0 if content == "yes" else 0.0

    first_token_info = logprobs_list[0]
    top_logprobs = first_token_info.get("top_logprobs", [])

    lp_yes: float | None = None
    lp_no: float | None = None

    for item in top_logprobs:
        token = item.get("token", "").strip().lower()
        logprob = item.get("logprob")
        if logprob is None:
            continue
        if token == "yes" and lp_yes is None:
            lp_yes = float(logprob)
        elif token == "no" and lp_no is None:
            lp_no = float(logprob)

    # Check generated token itself if not found in top_logprobs
    gen_token = first_token_info.get("token", "").strip().lower()
    gen_lp = first_token_info.get("logprob")
    if gen_lp is not None:
        if gen_token == "yes" and lp_yes is None:
            lp_yes = float(gen_lp)
        elif gen_token == "no" and lp_no is None:
            lp_no = float(gen_lp)

    # Handle cases where one token is absent from top-k
    if lp_yes is not None and lp_no is None:
        lp_no = -20.0
    elif lp_no is not None and lp_yes is None:
        lp_yes = -20.0
    elif lp_yes is None and lp_no is None:
        content = (
            response_data.get("message", {}).get("content", "").strip().lower()
        )
        return 1.0 if content == "yes" else 0.0

    return compute_binary_softmax(lp_yes, lp_no)


class BaseReranker(ABC):
    """Abstract base class for all RAG rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_n: int = 5,
    ) -> list[NodeWithScore]:
        """Reranks candidate nodes with respect to query.

        Args:
            query: The search query string.
            nodes: Initial candidate nodes with retrieval scores.
            top_n: Maximum number of top reranked nodes to return.

        Returns:
            Reranked list of NodeWithScore objects sorted descending by score.
        """
        pass

    def unload(self) -> None:
        """Unloads model resources from memory / VRAM."""
        pass


class NoneReranker(BaseReranker):
    """Identity reranker representing pure retrieval baseline."""

    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_n: int = 5,
    ) -> list[NodeWithScore]:
        """Returns the top_n candidate nodes in original retrieval order."""
        return nodes[:top_n]

    def unload(self) -> None:
        """No-op for baseline."""
        pass


class Qwen3OllamaReranker(BaseReranker):
    """Generative reranker using Qwen3 in Ollama with logprob softmax scoring."""

    DEFAULT_SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on "
        "the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
    )

    def __init__(
        self,
        model_name: str = "dengcao/Qwen3-Reranker-4B:Q4_K_M",
        base_url: str = "http://localhost:11434",
        instruction: str = "Retrieve documents relevant to the query",
        timeout: float = 30.0,
        keep_alive: int | str = "5m",
    ) -> None:
        """Initializes the Qwen3 Ollama Reranker.

        Args:
            model_name: Name of the Ollama model.
            base_url: Ollama base server URL.
            instruction: Instruction prompt template parameter.
            timeout: HTTP request timeout in seconds.
            keep_alive: Model memory retention in Ollama (e.g. '5m' or 0).
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.instruction = instruction
        self.timeout = timeout
        self.keep_alive = keep_alive

    def _score_document(self, query: str, document_text: str) -> float:
        """Calls Ollama /api/chat to score document relevancy."""
        user_prompt = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document_text}"
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "num_predict": 1,
                "temperature": 0,
            },
            "think": False,
            "logprobs": True,
            "top_logprobs": 20,
            "stream": False,
            "keep_alive": self.keep_alive,
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return parse_ollama_logprob_score(resp.json())

    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_n: int = 5,
    ) -> list[NodeWithScore]:
        """Reranks nodes using Ollama chat logprobs with graceful fallback."""
        if not nodes:
            return []

        try:
            reranked_nodes: list[NodeWithScore] = []
            for node in nodes:
                doc_text = node.node.get_content()
                score = self._score_document(query, doc_text)
                reranked_nodes.append(
                    NodeWithScore(node=node.node, score=score)
                )

            reranked_nodes.sort(
                key=lambda n: n.score if n.score is not None else float("-inf"),
                reverse=True,
            )
            return reranked_nodes[:top_n]
        except Exception as e:
            logger.warning(
                "Qwen3OllamaReranker failed (%s); fallback to original order.", e
            )
            return nodes[:top_n]

    def unload(self) -> None:
        """Unloads the model from Ollama GPU VRAM."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass


class CrossEncoderReranker(BaseReranker):
    """Discriminative cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        max_length: int = 512,
        model: Any = None,
    ) -> None:
        """Initializes CrossEncoderReranker.

        Args:
            model_name: HuggingFace model path or identifier.
            device: Computation device ('cpu', 'cuda').
            max_length: Maximum sequence length.
            model: Optional pre-instantiated model instance for testing.
        """
        self.model_name = model_name
        self.max_length = max_length
        self._model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model(self) -> Any:
        """Lazily instantiates the underlying CrossEncoder."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
        return self._model

    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_n: int = 5,
    ) -> list[NodeWithScore]:
        """Reranks nodes using CrossEncoder predict scores."""
        if not nodes:
            return []

        try:
            pairs = [[query, n.node.get_content()] for n in nodes]
            scores = self.model.predict(pairs)

            reranked_nodes = [
                NodeWithScore(node=nodes[i].node, score=float(scores[i]))
                for i in range(len(nodes))
            ]
            reranked_nodes.sort(
                key=lambda n: n.score if n.score is not None else float("-inf"),
                reverse=True,
            )
            return reranked_nodes[:top_n]
        except Exception as e:
            logger.warning(
                "CrossEncoderReranker failed (%s); fallback to original order.", e
            )
            return nodes[:top_n]

    def unload(self) -> None:
        """Unloads the model from PyTorch/CUDA VRAM."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RankGPTReranker(BaseReranker):
    """Listwise LLM reranker using LlamaIndex RankGPTRerank or listwise prompt."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        llm: Any = None,
        postprocessor: Any = None,
    ) -> None:
        """Initializes RankGPTReranker.

        Args:
            model_name: OpenAI model name.
            api_key: OpenAI API key.
            llm: Optional LlamaIndex LLM instance.
            postprocessor: Optional pre-configured postprocessor instance.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._llm = llm
        self._postprocessor = postprocessor

    @property
    def postprocessor(self) -> Any:
        """Lazily creates RankGPTRerank instance if available."""
        if self._postprocessor is not None:
            return self._postprocessor

        rankgpt_cls = None
        try:
            from llama_index.core.postprocessor.rankGPT_rerank import (
                RankGPTRerank as RankGPTCls,
            )

            rankgpt_cls = RankGPTCls
        except ImportError:
            try:
                from llama_index.postprocessor.rankgpt_rerank import (
                    RankGPTRerank as RankGPTCls,
                )

                rankgpt_cls = RankGPTCls
            except ImportError:
                pass

        if rankgpt_cls is not None:
            llm = self._llm
            if llm is None:
                from llama_index.llms.openai import OpenAI

                llm = OpenAI(model=self.model_name, api_key=self.api_key)
            self._postprocessor = rankgpt_cls(top_n=10, llm=llm)
            return self._postprocessor

        return None

    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_n: int = 5,
    ) -> list[NodeWithScore]:
        """Reranks nodes using RankGPTRerank or listwise prompt fallback."""
        if not nodes:
            return []

        try:
            pp = self.postprocessor
            if pp is not None:
                pp.top_n = top_n
                query_bundle = QueryBundle(query_str=query)
                reranked = pp.postprocess_nodes(nodes, query_bundle=query_bundle)
                return reranked[:top_n]

            return self._listwise_fallback(query, nodes, top_n)
        except Exception as e:
            logger.warning(
                "RankGPTReranker failed (%s); fallback to original order.", e
            )
            return nodes[:top_n]

    def _listwise_fallback(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_n: int = 5,
    ) -> list[NodeWithScore]:
        """Listwise ranking prompt fallback when postprocessor is unavailable."""
        from llama_index.llms.openai import OpenAI

        llm = self._llm or OpenAI(model=self.model_name, api_key=self.api_key)
        items_str = "\n".join(
            f"[{i+1}] {node.node.get_content()[:200]}"
            for i, node in enumerate(nodes)
        )
        prompt = (
            f"Given the query: '{query}', rank the following passages from "
            "most to least relevant.\n"
            "Respond with a comma-separated list of ranking numbers, e.g. [1, 2, 3].\n"
            f"Passages:\n{items_str}\n\nRanking:"
        )
        response = llm.complete(prompt).text
        numbers = [int(n) - 1 for n in re.findall(r"\d+", response)]
        valid_indices = [idx for idx in numbers if 0 <= idx < len(nodes)]
        remaining = [i for i in range(len(nodes)) if i not in valid_indices]
        ordered_indices = valid_indices + remaining
        return [nodes[i] for i in ordered_indices[:top_n]]

    def unload(self) -> None:
        """Cleans up internal LLM and postprocessor references."""
        self._postprocessor = None
        self._llm = None
