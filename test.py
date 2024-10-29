from util import import_ragas_custom, load_env_variables_from_all_env_files

import_ragas_custom('ragas_custom_2')
load_env_variables_from_all_env_files()
import os
import asyncio
import nest_asyncio

from llama_index.core import SimpleDirectoryReader
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from ragas.run_config import RunConfig
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.engine import Parallel
from ragas.prompt import PromptMixin
nest_asyncio.apply()
DATA = 'test'
LANGUAGE = "portuguese"
TIMEOUT = 2400.0
CACHE_DIR = 'cache_2'
AMOUNT_TESTS = 128
MODEL = 'gpt-4o-mini-2024-07-18'
run_config = RunConfig()
llm = LlamaIndexLLMWrapper(OpenAI(model=MODEL), run_config=run_config)
embedding = LlamaIndexEmbeddingsWrapper(OpenAIEmbedding(model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL), run_config=run_config)
generator = TestsetGenerator(llm)
query_distribution = default_query_distribution(llm)

transforms = default_transforms(
                llm=llm,
                embedding_model=embedding,
            )
list_transforms = [i for i, _ in query_distribution]

for i in transforms:
    if isinstance(i, Parallel):
        list_transforms.extend(i.transformations)
    else:
        list_transforms.append(i)

list_transforms = [i for i in list_transforms if isinstance(i, PromptMixin)]

for query in list_transforms:
    path = os.path.join(CACHE_DIR, query.__class__.__name__)
    if not os.path.exists(path):
        os.makedirs(path)

    try:
        prompts = query.load_prompts(path, LANGUAGE)
        query.set_prompts(**prompts)
    except Exception:
        prompts = asyncio.run(query.adapt_prompts(LANGUAGE, None, True, True))
        query.set_prompts(**prompts)
        query.save_prompts(path)
        prompts = query.load_prompts(path, LANGUAGE)
        query.set_prompts(**prompts)

documents = SimpleDirectoryReader(DATA).load_data()
testset = generator.generate_with_llamaindex_docs(documents, AMOUNT_TESTS, query_distribution=query_distribution,
                                                  with_debugging_logs=True, run_config=run_config, 
                                                  transforms_llm=llm,
                                                  transforms_embedding_model=embedding,
                                                  transforms=transforms)
testset.to_jsonl('testset_openai_4omini.jsonl')