import datasets
import json
from util import import_ragas_custom, load_env_variables_from_all_env_files

import_ragas_custom('ragas_custom_2')


from ragas.testset.synthesizers.testset_schema import Testset

test = Testset.from_jsonl('testset_openai_4omini.jsonl')

dataset = test.to_hf_dataset()

t = datasets.Dataset.from_list(dataset['eval_sample'])
print(t)
