from typing import Any, Dict, get_type_hints, List, Union
from ragas.metrics._faithfulness import LONG_FORM_ANSWER_PROMPT
import json

def generate_json_structure(cls: Any) -> Dict:
    def generate_json_structure_aux(cls: Any, depth: int = 1) -> Dict:
        def generate_example(field_type: Any, depth: int) -> Any:
            if field_type == str:
                return "string"
            elif field_type == float:
                return 0.00
            elif field_type == bool:
                return True
            elif field_type == int:
                return 0
            elif hasattr(field_type, '__origin__') and field_type.__origin__ == list:
                inner_type = field_type.__args__[0]
                return [generate_example(inner_type, depth + 1) for _ in range(depth)]
            elif hasattr(field_type, '__origin__') and field_type.__origin__ == dict:
                key_type, value_type = field_type.__args__
                return {generate_example(key_type, depth + 1): generate_example(value_type, depth + 1)}
            elif isinstance(field_type, type):
                return generate_json_structure_aux(field_type, depth + 1)
            else:
                return "undefined"

        json_structure = {}
        type_hints = get_type_hints(cls)
        
        for field, field_type in type_hints.items():
            if '__' not in field:
                json_structure[field] = generate_example(field_type, depth)
        
        return json_structure

    json_data = generate_json_structure_aux(cls)
    return json.dumps(json_data)

print(LONG_FORM_ANSWER_PROMPT.format(question="What is the capital of India?", answer="New Delhi", sentences="New Delhi is the capital of India.").to_string())