import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)

def _process_nodes(nodes):
    response = []
    for i in nodes:
        metadata = {k: str(v) for k, v in i.metadata.items()}
        response.append((i.text, metadata, i.node.get_node_info()))

    return response

def achat(queue_in, queue_out, persist_dir_received: str, openai_model: str, openai_embedding_model: str, api_key_openai: str) -> None:
    Settings.embed_model = OpenAIEmbedding(model=openai_embedding_model, api_key=api_key_openai)
    Settings.llm = OpenAI(model=openai_model, api_key=api_key_openai)
    
    persist_dir = persist_dir_received

    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        index = VectorStoreIndex.from_documents([])

    chat_engine = index.as_chat_engine()
    
    with open("achat.txt", "a") as f:
        f.write("achat\n")

    while True:
        message = queue_in.get()

        with open("achat.txt", "a") as f:
            f.write("achat t\n")
        if message is None:
            break
        
        aux_reponse = chat_engine.chat(message)
        queue_out.put((aux_reponse.response, _process_nodes(aux_reponse.source_nodes)))