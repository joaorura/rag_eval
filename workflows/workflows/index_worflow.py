from logging import getLogger

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step
)

from llama_index.core import VectorStoreIndex

logger = getLogger(__name__)

class IndexWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.index_retriever = index.as_retriever()

    @staticmethod
    def _process_nodes(nodes):
        response = []
        for i in nodes:
            metadata = {k: str(v) for k, v in i.metadata.items()}
            response.append((i.text, metadata, i.node.get_node_info()))
            
        return response
 
    @step
    async def get_index(self, ev: StartEvent) -> StopEvent:
        message = ev.get('message', None)
        
        if message is None:
            return StopEvent(result=None)
        
        response = await self.index_retriever.aretrieve(message)
        return StopEvent(result=self._process_nodes(response))

