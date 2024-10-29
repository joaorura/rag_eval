from ragas.testset.synthesizers.testset_schema import Testset


data = Testset.from_jsonl('testset_openai_4omini.jsonl')

print(data.to_list()[4])