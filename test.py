from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.testset.prompts import context_scoring_prompt, translate_all
data = 'teste'
language = "pt"
distributions = {
    simple:0.4,
    reasoning:0.2,
    multi_context:0.2,
    conditional:0.2
    }

translate_all(language, 'dsaas')