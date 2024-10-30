from util import import_ragas_custom, load_env_variables_from_all_env_files

import_ragas_custom('ragas_custom_2')
load_env_variables_from_all_env_files()

from ragas.testset.synthesizers.testset_schema import Testset


data = Testset.from_jsonl('testset_openai_4omini.jsonl')

print(data.to_list()[4])