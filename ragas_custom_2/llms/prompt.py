from __future__ import annotations

import json
import logging
import os
import typing as t

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue as BasePromptValue
from langchain_core.pydantic_v1 import BaseModel, root_validator
from deep_translator import GoogleTranslator

from ragas.llms import BaseRagasLLM
from ragas.utils import get_cache_dir
from ragas.utils import RAGAS_SUPPORTED_LANGUAGE_CODES

Example = t.Dict[str, t.Any]

logger = logging.getLogger(__name__)

def _check_if_language_is_supported(language: str):
    if language not in RAGAS_SUPPORTED_LANGUAGE_CODES:
        raise ValueError(
            f"Language '{language}' not supported. Supported languages: {RAGAS_SUPPORTED_LANGUAGE_CODES.keys()}"
        )


class PromptValue(BasePromptValue):
    prompt_str: str

    def to_messages(self) -> t.List[BaseMessage]:
        """Return prompt as a list of Messages."""
        return [HumanMessage(content=self.to_string())]

    def to_string(self) -> str:
        return self.prompt_str


class Prompt(BaseModel):
    """
    Prompt is a class that represents a prompt for the ragas metrics.

    Attributes:
        name (str): The name of the prompt.
        instruction (str): The instruction for the prompt.
        output_format_instruction (str): The output format instruction for the prompt.
        examples (List[Dict[str, Any]]): List of example inputs and outputs for the prompt.
        input_keys (List[str]): List of input variable names.
        output_key (str): The output variable name.
        output_type (Literal["json", "str"]): The type of the output (default: "json").
        language (str): The language of the prompt (default: "english").
    """

    name: str = ""
    instruction: str
    output_format_instruction: str = ""
    examples: t.List[Example] = []
    input_keys: t.List[str] = [""]
    output_key: str = ""
    output_type: t.Literal["json", "str"] = "json"
    language: str = "english"

    @root_validator
    def validate_prompt(cls, values: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """
        Validate the template string to ensure that it is in desired format.
        """
        if values.get("instruction") is None or values.get("instruction") == "":
            raise ValueError("instruction cannot be empty")
        if values.get("input_keys") is None or values.get("instruction") == []:
            raise ValueError("input_keys cannot be empty")
        if values.get("output_key") is None or values.get("output_key") == "":
            raise ValueError("output_key cannot be empty")

        if values.get("examples"):
            output_key = values["output_key"]
            for no, example in enumerate(values["examples"]):
                for inp_key in values["input_keys"]:
                    if inp_key not in example:
                        raise ValueError(
                            f"example {no+1} does not have the variable {inp_key} in the definition"
                        )
                if output_key not in example:
                    raise ValueError(
                        f"example {no+1} does not have the variable {output_key} in the definition"
                    )
                if values["output_type"].lower() == "json":
                    try:
                        if output_key in example:
                            if isinstance(example[output_key], str):
                                json.loads(example[output_key])
                    except ValueError as e:
                        raise ValueError(
                            f"{output_key} in example {no+1} is not in valid json format: {e}"
                        )

        return values

    def to_string(self) -> str:
        """
        Generate the prompt string from the variables.
        """
        prompt_elements = [self.instruction]
        if self.output_format_instruction:
            prompt_elements.append(
                "\n"
                + self.output_format_instruction.replace("{", "{{").replace("}", "}}")
            )
        prompt_str = "\n".join(prompt_elements) + "\n"

        if self.examples:
            prompt_str += "\nExamples:\n"
            # Format the examples to match the Langchain prompt template
            for example in self.examples:
                for key, value in example.items():
                    is_json = isinstance(value, (dict, list))
                    value = (
                        json.dumps(value, ensure_ascii=False).encode("utf8").decode()
                    )
                    value = (
                        value.replace("{", "{{").replace("}", "}}")
                        if self.output_type.lower() == "json"
                        else value
                    )
                    prompt_str += (
                        f"\n{key}: {value}"
                        if not is_json
                        else f"\n{key}: ```{value}```"
                    )
                prompt_str += "\n"

        prompt_str += "\nYour actual task:\n"

        if self.input_keys:
            prompt_str += "".join(f"\n{key}: {{{key}}}" for key in self.input_keys)
        if self.output_key:
            prompt_str += f"\n{self.output_key}: \n"

        return prompt_str

    def get_example_str(self, example_no: int) -> str:
        """
        Get the example string from the example number.
        """
        if example_no >= len(self.examples):
            raise ValueError(f"example number {example_no} is out of range")
        example = self.examples[example_no]
        example_str = ""
        for key, value in example.items():
            value = json.dumps(value, ensure_ascii=False).encode("utf8").decode()
            value = (
                value.replace("{", "{{").replace("}", "}}")
                if self.output_type.lower() == "json"
                else value
            )
            example_str += f"\n{key}: {value}"
        return "```" + example_str + "```"

    def format(self, **kwargs: t.Any) -> PromptValue:
        """
        Format the Prompt object into a ChatPromptTemplate object to be used in metrics.
        """
        if set(self.input_keys) != set(kwargs.keys()):
            raise ValueError(
                f"Input variables {self.input_keys} do not match with the given parameters {list(kwargs.keys())}"
            )
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs[key] = (
                    json.dumps(value, ensure_ascii=False).encode("utf8").decode()
                )

        prompt = self.to_string()
        return PromptValue(prompt_str=prompt.format(**kwargs))
    
    
    @staticmethod
    def translate_dict(data, language):
        translator = GoogleTranslator(source='en', target=language)
        
        def translate_aux(data_temp):
            if isinstance(data_temp, dict):
                for key, value in data_temp.items():
                    if key == "keyphrases" and isinstance(value, list):
                        data_temp[key] = [translator.translate(item) for item in value if isinstance(item, str)]
                    elif isinstance(value, str):
                        data_temp[key] = translator.translate(value)
                    elif isinstance(value, list):
                        data_temp[key] = [translate_aux(item) for item in value]
                    elif isinstance(value, dict):
                        data_temp[key] = translate_aux(value)
            elif isinstance(data_temp, list):
                data_temp = [translate_aux(item) for item in data_temp]
                
            return data_temp

        return translate_aux(data)
    
    def adapt(
        self, language: str, llm: BaseRagasLLM, cache_dir: t.Optional[str] = None
    ) -> Prompt:
        _check_if_language_is_supported(language)

        def get_all_keys(nested_json):
            keys = set()
            for key, value in nested_json.items():
                keys.add(key)
                if isinstance(value, dict):
                    keys = keys.union(get_all_keys(value))
            return keys

        if self.language == language:
            return self

        # TODO: Add callbacks
        cache_dir = cache_dir if cache_dir else get_cache_dir()
        if os.path.exists(os.path.join(cache_dir, language, f"{self.name}.json")):
            self_cp_dict = self._load(language, self.name, cache_dir).dict()

            for key in self_cp_dict.keys():
                setattr(self, key, self_cp_dict[key])

            return self

        logger.info("Adapting %s to %s", self.name, language)

        dict_var = self.dict()
        removed_keys = ['name', 'input_keys', 'output_key', 'output_type', 'language']

        for key in removed_keys:
            del dict_var[key]

        translated_dict = Prompt.translate_dict(dict_var, language)
        
        for key in translated_dict.keys():
            setattr(self, key, translated_dict[key])

        self.language = language

        self.save(cache_dir=cache_dir)

        return self

    def save(self, cache_dir: t.Optional[str] = None):
        cache_dir = cache_dir if cache_dir else get_cache_dir()
        cache_dir = os.path.join(cache_dir, self.language)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        cache_path = os.path.join(cache_dir, f"{self.name}.json")
        with open(cache_path, "w") as file:
            json.dump(self.dict(), file, indent=4)

    @classmethod
    def _load(cls, language: str, name: str, cache_dir: str) -> Prompt:
        logger.info("Loading %s from %s", name, cache_dir)
        path = os.path.join(cache_dir, language, f"{name}.json")
        return cls(**json.load(open(path)))


str_translation = Prompt(
    name="str_translation",
    instruction="Language translation",
    examples=[
        {
            "translate_to": "hindi",
            "input": "Who was  Albert Einstein and what is he best known for?",
            "output": "अल्बर्ट आइंस्टीन कौन थे और वे किसके लिए सबसे ज्यादा प्रसिद्ध हैं?",
        },
        {
            "translate_to": "dutch",
            "input": "Who was queen Elizabeth and what is she best known for?",
            "output": "Wie was koningin Elizabeth en waar is zij het meest bekend om?",
        },
    ],
    input_keys=["translate_to", "input"],
    output_key="output",
    output_type="str",
)

json_translatation = Prompt(
    name="json_translation",
    instruction="Translate values in given json to target language and output the translated json",
    examples=[
        {
            "translate_to": "hindi",
            "input": {
                "statements": [
                    "Albert Einstein was born in Germany.",
                    "Albert Einstein was best known for his theory of relativity.",
                ]
            },
            "output": {
                "statements": [
                    "अल्बर्ट आइंस्टीन का जन्म जर्मनी में हुआ था।",
                    "अल्बर्ट आइंस्टीन अपने सापेक्षता के सिद्धांत के लिए सबसे अधिक प्रसिद्ध थे।",
                ]
            },
        },
        {
            "translate_to": "dutch",
            "input": {
                "statements": [
                    "Paris is the capital of France.",
                    "Croissants are a popular French pastry.",
                ]
            },
            "output": {
                "statements": [
                    "Parijs is de hoofdstad van Frankrijk.",
                    "Croissants zijn een populair Frans gebak.",
                ]
            },
        },
    ],
    input_keys=["translate_to", "input"],
    output_key="output",
    output_type="json",
)
