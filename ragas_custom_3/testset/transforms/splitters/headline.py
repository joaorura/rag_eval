import re

import typing as t

from dataclasses import dataclass
from unidecode import unidecode

from ragas.testset.graph import Node, NodeType, Relationship
from ragas.testset.transforms.base import Splitter


def normalize_text(text):
    return unidecode(re.sub(r'\s+', '', text).lower())


def remove_indices(text):
    cleaned_text = re.sub(r'(\d+\.)+ *', '', text)
    return cleaned_text


def adjust_indices(original_text, indices):
    last_index = 0
    count = 0

    indices = sorted(indices)
    new_indices = []
    for index in indices:
        while last_index < len(original_text):
            if not original_text[last_index].isspace():
                count += 1
            if count == index + 1:
                new_indices.append(last_index)
                last_index += 1
                break
            last_index += 1
    
    return new_indices

@dataclass
class HeadlineSplitter(Splitter):
    min_tokens: int = 300
    max_tokens: int = 1000

    def adjust_chunks(self, chunks):
        adjusted_chunks = []
        current_chunk = ""

        for chunk in chunks:
            chunk_tokens = chunk.split()

            # Split chunks that are over max_tokens
            while len(chunk_tokens) > self.max_tokens:
                adjusted_chunks.append(" ".join(chunk_tokens[: self.max_tokens]))
                chunk_tokens = chunk_tokens[self.max_tokens :]

            # Handle chunks that are under min_tokens
            if len(chunk_tokens) < self.min_tokens:
                if current_chunk:
                    current_chunk += " " + " ".join(chunk_tokens)
                    if len(current_chunk.split()) >= self.min_tokens:
                        adjusted_chunks.append(current_chunk)
                        current_chunk = ""
                else:
                    current_chunk = " ".join(chunk_tokens)
            else:
                if current_chunk:
                    adjusted_chunks.append(current_chunk)
                    current_chunk = ""
                adjusted_chunks.append(" ".join(chunk_tokens))

        # Append any remaining chunk
        if current_chunk:
            adjusted_chunks.append(current_chunk)

        return adjusted_chunks

    async def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        text = node.get_property("page_content")
        if text is None:
            raise ValueError("'page_content' property not found in this node")

        headlines = node.get_property("headlines")
        if headlines is None:
            raise ValueError("'headlines' property not found in this node")

        if len(text.split()) < self.min_tokens:
            return [node], []
        # create the chunks for the different sections
        indices = [0]
        normalized_text = normalize_text(text)

        for headline in headlines:
            if headline is not None and not headline.isspace():
                indice = normalized_text.find(normalize_text(headline))
                if indice == -1:
                    text_search = remove_indices(headline)
                    text_search = normalize_text(text_search)
                    indice = normalized_text.find(text_search)

                if indice != -1:
                    indices.append(indice)
        
        indices.append(len(text))
        indices = adjust_indices(text, indices)

        chunks = [text[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]
        chunks = self.adjust_chunks(chunks)

        # if there was no headline, return the original node
        if len(chunks) == 1:
            return [node], []

        # create the nodes
        nodes = [
            Node(type=NodeType.CHUNK, properties={"page_content": chunk})
            for chunk in chunks
        ]

        # create the relationships for children
        relationships = []
        for child_node in nodes:
            relationships.append(
                Relationship(
                    type="child",
                    source=node,
                    target=child_node,
                )
            )

        # create the relationships for the next nodes
        for i, child_node in enumerate(nodes):
            if i < len(nodes) - 1:
                relationships.append(
                    Relationship(
                        type="next",
                        source=child_node,
                        target=nodes[i + 1],
                    )
                )
        return nodes, relationships
