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
from ragas.run_config import RunConfig
from ragas.llms import LlamaIndexLLMWrapper
nest_asyncio.apply()
DATA = 'test'
LANGUAGE = "portuguese"
TIMEOUT = 10.0
CACHE_DIR = 'cache_2'
AMOUNT_TESTS = 2
MODEL = 'gpt-4o-mini-2024-07-18'
run_config = RunConfig(max_workers=1)
llm = LlamaIndexLLMWrapper(OpenAI(model=MODEL), run_config=run_config)

generator = TestsetGenerator(llm)
query_distribution = default_query_distribution(llm)
for query, _ in query_distribution[:1]:
    path = os.path.join(CACHE_DIR, query.__class__.__name__)
    if not os.path.exists(path):
        os.makedirs(path)

    try:
        prompts = query.load_prompts(path, LANGUAGE)
        query.set_prompts(**prompts)
    except Exception:
        prompts = asyncio.run(query.adapt_prompts(LANGUAGE, llm, True))
        query.set_prompts(**prompts)
        query.save_prompts(path)
        prompts = query.load_prompts(path, LANGUAGE)
        query.set_prompts(**prompts)
