from logging import getLogger

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step
)

from llama_index.core import VectorStoreIndex

logger = getLogger(__name__)
   
class QueryWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.query_engine = index.as_query_engine()
    
    @staticmethod
    def _process_nodes(nodes):
        response = []
        for i in nodes:
            metadata = {k: str(v) for k, v in i.metadata.items()}
            response.append((i.text, metadata, i.node.get_node_info()))
            
        return response

    @step
    async def query(self, ev: StartEvent) -> StopEvent:
        message = ev.get('message', None)
        
        if message is None:
            return StopEvent(result=None)
        
        stream_response = await self.query_engine.aquery(message)
        return StopEvent(result=(stream_response.response, self._process_nodes(stream_response.source_nodes)))


