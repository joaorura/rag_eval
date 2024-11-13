import os

from .chat_workflow import ChatWorkflow
from .index_worflow import IndexWorkflow
from .query_workflow import QueryWorkflow
from .manage_workflow import ManageWorkflow

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

Settings.embed_model = OpenAIEmbedding(model=os.environ.get('MODEL_EMBEDDING'))
Settings.llm = OpenAI(model=os.environ.get('MODEL_LLM'))

qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
client = QdrantClient(host=qdrant_host, port=6333)
aclient = AsyncQdrantClient(host=qdrant_host, port=6333)
vector_store = QdrantVectorStore(
    collection_name="docs",
    client=client,
    aclient=aclient,
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=Settings.embed_model
)

if not client.collection_exists("docs"):
    nodes = SimpleDirectoryReader("/data").load_data()

    for transform in Settings.transformations:
        nodes = transform(nodes)

    index.insert_nodes(nodes)

os.makedirs("/tmp", exist_ok=True)

chat_w = ChatWorkflow(index=index)
index_w = IndexWorkflow(index=index)
query_w = QueryWorkflow(index=index)
manage_w = ManageWorkflow(index=index)