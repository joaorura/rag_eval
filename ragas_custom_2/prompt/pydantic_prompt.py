from __future__ import annotations

import copy
import json
import logging
import os
import hashlib

import typing as t

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from deep_translator import GoogleTranslator

from ragas._version import __version__
from ragas.callbacks import new_group
from ragas.exceptions import RagasOutputParserException
from ragas.llms.prompt import PromptValue

from .base import BasePrompt, StringIO, _check_if_language_is_supported

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.base import BaseRagasLLM

logger = logging.getLogger(__name__)

# type variables for input and output models
InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)

def hash_recursive(obj):
    hash_obj = hashlib.sha256()

    def _hash_recursive(obj):
        if isinstance(obj, (tuple, list, set)):
            for indice, value in enumerate(obj):
                hash_obj.update(str(indice).encode())
                hash_recursive(value)
        elif isinstance(obj, dict):
            for key, value in sorted(obj.items()):
                hash_obj.update(str(key).encode())
                hash_recursive(value)
        elif hasattr(obj, '__dict__'):
            for name in dir(obj):
                if not name.startswith('_'):
                    value = getattr(obj, name)
                    if not callable(value):
                        hash_obj.update(name.encode())
                        hash_recursive(value)
        else:
            hash_obj.update(str(obj).encode())

    _hash_recursive(obj)
    
    return int.from_bytes(hash_obj.digest(), 'big')

class PydanticPromptStrings:
    output_signature = (
        "Please return the output in a format that complies with the "
        "following schema as specified in JSON Schema and OpenAPI specification:"
    )
    examples_intro = "These are some examples to show how to perform the above instruction"
    instruction_prompt = "Now perform the above instruction with the following input"

    language = 'english'

    def adapt(self, target_language: str) -> PydanticPromptStrings:
        if self.language == target_language:
            return self
        
        data = copy.deepcopy(self)

        translator = GoogleTranslator(source='en', target=target_language)

        vars = dir(data)
        vars.remove('language')
        
        for name in vars:
            value = getattr(data, name)
            if not callable(value) and not name.startswith('_'):
                setattr(data, name, translator.translate(value))

        data.language = target_language

        return data

    def save(self, file_path: str):
        data = {}

        for name in dir(self):
            value = getattr(self, name)
            if not callable(value) and not name.startswith('_'):
                data[name] = value
        
        data['ragas_version'] = __version__
        if os.path.exists(file_path):
            raise FileExistsError(f"The file '{file_path}' already exists.")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Prompt strings saved to {file_path}")

    @staticmethod
    def load( file_path: str) -> PydanticPromptStrings:
        with open(file_path, "r") as f:
            data = json.load(f)

        prompt_strings = PydanticPromptStrings()

        ragas_version = data.pop("ragas_version")

        if ragas_version != __version__:
            logger.warning(
                "Prompt strings were saved with Ragas v%s, but you are loading it with Ragas v%s. "
                "There might be incompatibilities.",
                ragas_version,
                __version__,
            )
        
        for name, value in data.items():
            setattr(prompt_strings, name, value)

        return prompt_strings


class PydanticPrompt(BasePrompt, t.Generic[InputModel, OutputModel]):
    input_model: t.Type[InputModel]
    output_model: t.Type[OutputModel]
    instruction: str
    examples: t.List[t.Tuple[InputModel, OutputModel]] = []
    strings: PydanticPromptStrings = PydanticPromptStrings()

    def _generate_instruction(self) -> str:
        return self.instruction

    def _generate_output_signature(self, indent: int = 4) -> str:
        return (
            self.strings.output_signature + '\n'
            + str(self.output_model.model_json_schema())
        )

    def _generate_examples(self):
        if self.examples:
            example_strings = []
            for e in self.examples:
                input_data, output_data = e
                example_strings.append(
                    self.instruction
                    + "\n"
                    + "input: "
                    + input_data.model_dump_json(indent=4)
                    + "\n"
                    + "output: "
                    + output_data.model_dump_json(indent=4)
                )

            return (
                self.strings.examples_intro + '\n'
                + "\n\n".join(example_strings)
            )
        else:
            return ""

    def to_string(self, data: t.Optional[InputModel] = None) -> str:
        return (
            self._generate_instruction()
            + "\n"
            + self._generate_output_signature()
            + "\n"
            + self._generate_examples()
            + "\n"
            + self.strings.instruction_prompt + '\n'
            + (
                "input: " + data.model_dump_json(indent=4) + "\n"
                if data is not None
                else "input: (None)\n"
            )
            + "output: "
        )

    async def generate(
        self,
        llm: BaseRagasLLM,
        data: InputModel,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> OutputModel:
        callbacks = callbacks or []

        output_single = await self.generate_multiple(
            llm=llm,
            data=data,
            n=1,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        return output_single[0]

    async def generate_multiple(
        self,
        llm: BaseRagasLLM,
        data: InputModel,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> t.List[OutputModel]:
        callbacks = callbacks or []
        processed_data = self.process_input(data)
        prompt_rm, prompt_cb = new_group(
            name=self.name,
            inputs={"data": processed_data},
            callbacks=callbacks,
        )
        prompt_value = PromptValue(prompt_str=self.to_string(processed_data))
        resp = await llm.generate(
            prompt_value,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=prompt_cb,
        )

        output_models = []
        parser = RagasOutputParser(pydantic_object=self.output_model, language=self.language)
        for i in range(n):
            output_string = resp.generations[0][i].text
            try:
                answer = await parser.parse_output_string(
                    output_string=output_string,
                    prompt_value=prompt_value,
                    llm=llm,
                    callbacks=prompt_cb,
                    max_retries=3,
                )
                processed_output = self.process_output(answer, data)  # type: ignore
                output_models.append(processed_output)
            except RagasOutputParserException as e:
                prompt_rm.on_chain_error(error=e)
                logger.error("Prompt %s failed to parse output: %s", self.name, e)
                raise e

        prompt_rm.on_chain_end({"output": output_models})
        return output_models

    def process_input(self, input: InputModel) -> InputModel:
        return input

    def process_output(self, output: OutputModel, input: InputModel) -> OutputModel:
        return output

    async def adapt(
        self, target_language: str, llm: t.Optional[BaseRagasLLM] = None
    ) -> "PydanticPrompt[InputModel, OutputModel]":
        _check_if_language_is_supported(target_language)

        if self.language == target_language:
            return self
        
        if self.original_hash is None:
            self.original_hash = hash(self)

        translator = GoogleTranslator(source='en', target=target_language)

        def translate(obj):
            if isinstance(obj, dict):
                return {k: translate(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return type(obj)(translate(item) for item in obj)
            elif hasattr(obj, '__dict__'):
                obj = copy.deepcopy(obj)
                for k, v in obj.__dict__.items():
                    setattr(obj, k, translate(v))
                return obj
            elif isinstance(obj, str):
                return translator.translate(obj)
            else:
                return obj
        
        new_prompt = copy.deepcopy(self)
        new_prompt.strings = new_prompt.strings.adapt(target_language)

        vars = dir(new_prompt)
        vars.remove('strings')
        vars.remove('language')
        vars.remove('name')

        for name in vars:
            value = getattr(new_prompt, name)
            if not callable(value) and not name.startswith('_'):
                setattr(new_prompt, name, translate(value))

        new_prompt.language = target_language
        new_prompt.original_hash = hash(new_prompt)

        return new_prompt

    def __repr__(self):
        return f"{self.__class__.__name__}(instruction={self.instruction}, examples={self.examples}, language={self.language})"

    def __str__(self):
        json_str = json.dumps(
            {
                "name": self.name,
                "instruction": self.instruction,
                "examples": [
                    (e[0].model_dump(), e[1].model_dump()) for e in self.examples
                ],
                "language": self.language,
            },
            indent=2,
            ensure_ascii=False,
        )[1:-1]
        return f"{self.__class__.__name__}({json_str})"

    def __hash__(self):
        return hash_recursive(self)

    def __eq__(self, other):
        if not isinstance(other, PydanticPrompt):
            return False
        return (
            self.name == other.name
            and self.input_model == other.input_model
            and self.output_model == other.output_model
            and self.instruction == other.instruction
            and self.examples == other.examples
            and self.language == other.language
        )

    def save(self, file_path: str):
        if os.path.exists(file_path):
            raise FileExistsError(f"The file '{file_path}' already exists.")

        def _save(obj):
            if isinstance(obj, dict):
                return {k: _save(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return type(obj)(_save(item) for item in obj)
            elif hasattr(obj, '__dict__'):
                data = {k: _save(v) for k, v in obj.__dict__.items() if not k.startswith('_') and not callable(v)}
                return data
            else:
                return obj
        
        diretorio, _ = os.path.split(file_path)
        path_strings = os.path.join(diretorio, f'strings_{__version__}.json')

        if not os.path.exists(path_strings):
            self.strings.save(path_strings)

        data = {}
        vars = dir(self)
        vars.remove('strings')
        vars.remove('name')

        for name in vars:
            value = getattr(self, name)
            if not callable(value) and not name.startswith('_'):
                data[name] = _save(value)

        if self.original_hash is None:
            data['original_hash'] = hash(self)

        data["ragas_version"] = __version__

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
            print(f"Prompt saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "PydanticPrompt[InputModel, OutputModel]":
        with open(file_path, "r") as f:
            data = json.load(f)

        ragas_version = data.pop("ragas_version")
        if ragas_version != __version__:
            logger.warning(
                "Prompt was saved with Ragas v%s, but you are loading it with Ragas v%s. "
                "There might be incompatibilities.",
                ragas_version,
                __version__,
            )
        
        original_hash = data.pop("original_hash")

        examples = data.pop("examples")

        prompt = cls()
        for name, value in data.items():
            setattr(prompt, name, value)

        prompt.examples = [
            (
                prompt.input_model(**example[0]),
                prompt.output_model(**example[1]),
            )
            for example in examples
        ]
        
        diretorio, _ = os.path.split(file_path)
        path_strings = os.path.join(diretorio, f'strings_{__version__}.json')

        if os.path.exists(path_strings):
            prompt.strings = PydanticPromptStrings.load(path_strings)

        calculated_hash = hash(prompt)
        if original_hash is not None and calculated_hash != original_hash:
            logger.warning("Loaded prompt hash does not match the saved hash.")

        return prompt


# Ragas Output Parser
class OutputStringAndPrompt(BaseModel):
    output_string: str
    prompt_value: str


class FixOutputFormat(PydanticPrompt[OutputStringAndPrompt, StringIO]):
    instruction = "The output string did not satisfy the constraints given in the prompt. Fix the output string and return it."
    input_model = OutputStringAndPrompt
    output_model = StringIO


fix_output_format_prompt = FixOutputFormat()

class RagasOutputParser(PydanticOutputParser[OutputModel]):
    async def parse_output_string(
        self,
        output_string: str,
        prompt_value: PromptValue,
        llm: BaseRagasLLM,
        callbacks: Callbacks,
        max_retries: int = 1,
        language: str = 'english'
    ):
        callbacks = callbacks or []
        try:
            result = super().parse(output_string)
        except OutputParserException:
            if max_retries != 0:
                retry_rm, retry_cb = new_group(
                    name="fix_output_format",
                    inputs={"output_string": output_string},
                    callbacks=callbacks,
                )

                fix_output_format_prompt = fix_output_format_prompt.adapt(language)

                fixed_output_string = await fix_output_format_prompt.generate(
                    llm=llm,
                    data=OutputStringAndPrompt(
                        output_string=output_string,
                        prompt_value=prompt_value.to_string(),
                    ),
                    callbacks=retry_cb,
                )
                retry_rm.on_chain_end({"fixed_output_string": fixed_output_string})
                return await self.parse_output_string(
                    output_string=fixed_output_string.text,
                    prompt_value=prompt_value,
                    llm=llm,
                    max_retries=max_retries - 1,
                    callbacks=callbacks,
                )
            else:
                raise RagasOutputParserException(num_retries=max_retries)
        return result
