from util import import_ragas_custom, load_env_variables_from_all_env_files

import_ragas_custom('ragas_custom_2')
load_env_variables_from_all_env_files()
import os
import asyncio
import nest_asyncio

from ragas.integrations.llama_index import evaluate
from ragas.run_config import RunConfig
from ragas.testset.synthesizers.testset_schema import Testset
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from ragas.prompt.mixin import PromptMixin
from ragas.run_config import RunConfig

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from ragas.metrics import (
    context_precision,
    context_recall,
    context_entity_recall,
    NoiseSensitivity,
    ResponseRelevancy,
    answer_relevancy,
    faithfulness,
    FactualCorrectness,
    SemanticSimilarity,
    NonLLMStringSimilarity,
    BleuScore,
    RougeScore,
    StringPresence,
    ExactMatch
)

from ragas.metrics._aspect_critic import SUPPORTED_ASPECTS
nest_asyncio.apply()
DATA_PATH = 'test'
TESTSET = 'testset_openai_4omini.jsonl'
PERSIST_DIR = "./storage"
LANGUAGE = 'portuguese'
TIMEOUT = 2400
RESULT_CSV = 'result_gpt4mini_gpt4mini.csv'
MODEL = 'gpt-4o-mini-2024-07-18'
CACHE_DIR = 'cache_2'
run_config = RunConfig(timeout=TIMEOUT)
testset = Testset.from_jsonl(TESTSET).to_pandas()

print("Tamanho do dataset: ", len(testset))
print(testset.head())
nan_rows = testset[testset.isna().any(axis=1)]

print("Quantidade de nulos: ", len(nan_rows))
print(nan_rows)

del nan_rows
testset = testset.dropna()
print("Quantidade de linhas após remoção de nulos: ", len(testset))
print(testset)
testset = Testset.from_pandas(testset[:1])
embeding = OpenAIEmbedding(model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL)
model = OpenAI(model=MODEL)

Settings.embed_model = embeding
Settings.llm = model
metrics = [
    context_precision,
    context_recall,
    context_entity_recall,
    NoiseSensitivity(),
    ResponseRelevancy(),
    answer_relevancy,
    faithfulness,
    FactualCorrectness(),
    SemanticSimilarity(),
    NonLLMStringSimilarity(),
    BleuScore(),
    RougeScore(),
    StringPresence(),
    ExactMatch()
]

metrics.extend(SUPPORTED_ASPECTS)

for query in metrics:
    if isinstance(query, PromptMixin):
        path = os.path.join(CACHE_DIR, query.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path)

        try:
            prompts = query.load_prompts(path, LANGUAGE,)
            query.set_prompts(**prompts)
        except Exception:
            prompts = asyncio.run(query.adapt_prompts(LANGUAGE, None, True, True))
            query.set_prompts(**prompts)
            query.save_prompts(path)
            prompts = query.load_prompts(path, LANGUAGE)
            query.set_prompts(**prompts)

testset.to_list()
documents = SimpleDirectoryReader(DATA_PATH).load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
query_engine = index.as_query_engine(request_timeout=TIMEOUT)
result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=testset,
    llm=model,
    embeddings=embeding,
    run_config=run_config
)
result_dataframe = result.to_pandas()
result_dataframe.to_csv(RESULT_CSV)
print(result)