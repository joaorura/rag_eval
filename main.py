import multi
import os
from multiprocessing import Process, Queue
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)

chat_engines = {}
achat_engines = {}
persist_dir = None
index = None
chat_index = 0
achat_index = 0
index_retriever = None

def create_chat() -> int:
    global chat_engines, chat_index, index
    chat_index += 1
    chat_engines[chat_index] = index.as_chat_engine()
    return chat_index

def remove_chat(index_chat: int) -> None:
    global chat_engines
    chat_engines.pop(index, None)
 
def _process_nodes(nodes):
    response = []
    for i in nodes:
        metadata = {k: str(v) for k, v in i.metadata.items()}
        response.append((i.text, metadata, i.node.get_node_info()))

    return response

def chat(index_chat: int, message: str) -> tuple[str, list[tuple[str, dict[str, str], dict[str, int]]]]:
    global chat_engines
    aux_reponse = chat_engines[index_chat].chat(message)

    return (aux_reponse.response, _process_nodes(aux_reponse.source_nodes))

def start(persist_dir_received: str, openai_model_: str, openai_embedding_model_: str, api_key_openai_: str) -> None:
    global persist_dir, index, index_retriever, openai_model, openai_embedding_model, api_key_openai
    
    openai_model = openai_model_
    openai_embedding_model = openai_embedding_model_
    api_key_openai = api_key_openai_

    Settings.embed_model = OpenAIEmbedding(model=openai_embedding_model, api_key=api_key_openai)
    Settings.llm = OpenAI(model=openai_model, api_key=api_key_openai)
    
    persist_dir = persist_dir_received

    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        index = VectorStoreIndex.from_documents([])

    index_retriever = index.as_retriever()

def add(doc_path: str) -> None:
    documents = SimpleDirectoryReader(doc_path).load_data()
    
    for i in documents:
        index.insert(i)
    
    index.storage_context.persist(persist_dir=persist_dir)

def remove(doc_path: str) -> None:
    documents = SimpleDirectoryReader(doc_path).load_data()
    index.remove_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)

def get_index(query: str) -> list[tuple[str, dict[str, str], dict[str, int]]]:
    global index_retriever
    aux = index_retriever.retrieve(query)
    return _process_nodes(aux)

def test():
    with open("achat.txt", "a") as f:
        f.write("teste\n")

def create_achat() -> int:
    global achat_engines, achat_index, index, openai_model, openai_embedding_model, api_key_openai, persist_dir
    
    queue_in = Queue()
    queue_out = Queue()
    
    Process(target=test).start()

    Process(target=multi.achat, args=(queue_in, queue_out, persist_dir, openai_model, openai_embedding_model, api_key_openai)).start()

    achat_index += 1
    achat_engines[achat_index] = (queue_in, queue_out)
    return achat_index

def remove_achat(index_chat: int) -> None:
    global achat_engines

    if index_chat not in achat_engines:
        return
    achat_engines[index_chat][0].put(None)
    achat_engines.pop(index_chat, None)

def achat(index_chat: int, message: str) -> None:
    global achat_engines
    
    if index_chat not in achat_engines:
        return

    achat_engines[index_chat][0].put(message)

def get_achat(index_chat: int) -> tuple[str, list[tuple[str, dict[str, str], dict[str, int]]]]:
    global achat_engines
    
    if index_chat not in achat_engines or achat_engines[index_chat][1].empty():
        return ("-1", [])

    return achat_engines[index_chat][1].get()