# %%
from util import import_ragas_custom, load_env_variables_from_all_env_files

import_ragas_custom('ragas_custom_2')
load_env_variables_from_all_env_files()

# %%
import os
import asyncio
import nest_asyncio

from ragas.integrations.llama_index import evaluate
from ragas.run_config import RunConfig
from ragas.testset.synthesizers.testset_schema import Testset
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.llms.llama_api import LlamaAPI

from ragas.prompt.mixin import PromptMixin
from ragas.run_config import RunConfig

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
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
    RougeScore,
    StringPresence,
    ExactMatch
)

from ragas.metrics._aspect_critic import SUPPORTED_ASPECTS

# %%
nest_asyncio.apply()

# %%
DATA_PATH = 'data'
TESTSET = 'testset_openai_4omini.jsonl'
LANGUAGE = 'portuguese'
TIMEOUT = 300
RESULT_CSV = 'result_gpt4omini_gemma2-9b'
MODEL_GPT = 'gpt-4o-mini-2024-07-18'
MODEL_GPT_EMBEDING = OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
CACHE_DIR = 'cache'
PERSIST_DIR = "./cache_gpt_emb_small"

# %%
run_config = RunConfig(timeout=TIMEOUT, max_workers=1)

# %%
testset = Testset.from_jsonl(TESTSET).to_pandas()

print("Tamanho do dataset: ", len(testset))
print(testset.head())

# %%
nan_rows = testset[testset.isna().any(axis=1)]

print("Quantidade de nulos: ", len(nan_rows))
print(nan_rows)

del nan_rows

# %%
testset = testset.dropna()

# %%
print("Quantidade de linhas após remoção de nulos: ", len(testset))
print(testset)

# %%
base = (len(testset) / 5)
index_csv= 1
min = int((index_csv - 1) * base )
max = int(index_csv * base)
testset = Testset.from_pandas(testset[min:max])
print(f'min: {min} | max: {max}')

# %%
test = {
    'llama3.2_1b': LlamaAPI(model='llama3.2-1b', api_key=os.getenv('LLAMA_API_KEY')),
    'llama3.2_3b': LlamaAPI(model='llama3.2-3b', api_key=os.getenv('LLAMA_API_KEY')),
    'gpt-4o-mini-2024-07-18': OpenAI(model='gpt-4o-mini-2024-07-18', api_key=os.getenv('OPENAI_API_KEY')),
    'gemma2-9b': LlamaAPI(model='gemma2-9b', api_key=os.getenv('LLAMA_API_KEY')),
}

# %%

model = test['gemma2-9b']
avaliator_llm = OpenAI(model=MODEL_GPT)
embeding = OpenAIEmbedding(model=MODEL_GPT_EMBEDING)


Settings.embed_model = embeding
Settings.llm = model

# %%
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


# %%
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    
query_engine = index.as_query_engine(request_timeout=TIMEOUT)

# %%
model.complete("test")

# %%
query_engine.query("teste")

# %%
asyncio.run(query_engine.aquery("teste"))

# %%
result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=testset,
    llm=avaliator_llm,
    embeddings=embeding,
    run_config=run_config
)

# %%
result_dataframe = result.to_pandas()
result_dataframe.to_csv(f'{RESULT_CSV}_{6}.csv')

# %%
print(result)


