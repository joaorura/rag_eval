import json
import logging
import typing as t

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from ragas.llms import BaseRagasLLM
from ragas.llms.prompt import Prompt, PromptValue

logger = logging.getLogger(__name__)
# The get_format_instructions function is a modified version from
# langchain_core.output_parser.pydantic. The original version removed the "type" json schema
# property that confused some older LLMs.

TBaseModel = t.TypeVar("TBaseModel", bound=BaseModel)

FIX_OUTPUT_FORMAT = Prompt(
    name="",
    instruction="Below, the Completion did not satisfy the constraints given in the Prompt.",
    output_format_instruction="",
    input_keys=["prompt", "completion"],
    output_key="fixed_completion",
)


JSON_FORMAT_INSTRUCTIONS = """The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"motive": "string", "note": 0}} the object {{"motive": "Knife", "note": 7}}.

Here is the output JSON schema:
```
{schema}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```)."""

def generate_json_structure(cls: t.Any) -> t.Dict:
    def generate_json_structure_aux(cls: t.Any, depth: int = 1) -> t.Dict:
        def generate_example(field_type: t.Any, depth: int) -> t.Any:
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
        type_hints = t.get_type_hints(cls)
        
        for field, field_type in type_hints.items():
            if field == '__root__':
                return generate_example(field_type, depth)
            
            if '__' not in field:
                json_structure[field] = generate_example(field_type, depth)
        
        return json_structure

    json_data = generate_json_structure_aux(cls)
    return json.dumps(json_data)

def get_json_format_instructions(pydantic_object) -> str:
    schema_str = generate_json_structure(pydantic_object)
    resp = JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)
    return resp

class RagasOutputParserOld(PydanticOutputParser):
    async def aparse(  # type: ignore
        self, result: str, prompt: PromptValue, llm: BaseRagasLLM, max_retries: int = 1
    ):
        try:
            output = super().parse(result)
        except OutputParserException:
            if max_retries != 0:
                p_value = FIX_OUTPUT_FORMAT.format(
                    prompt=prompt.to_string(), completion=result
                )
                output = await llm.generate(p_value)
                result = output.generations[0][0].text
                return await self.aparse(result, prompt, llm, max_retries - 1)
            else:
                logger.warning("Failed to parse output. Returning None.")
                return None
        return output
