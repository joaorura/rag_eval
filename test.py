from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.testset.prompts import filter_question_prompt
from ragas.llms.prompt import str_translation, json_translatation

data = 'teste'
language = "Português"
distributions = {
    simple:0.4,
    reasoning:0.2,
    multi_context:0.2,
    conditional:0.2
    }

filter_question_prompt.adapt(language, LangchainLLMWrapper(ChatOllama(model='llama3.2')))


print(filter_question_prompt.to_string())