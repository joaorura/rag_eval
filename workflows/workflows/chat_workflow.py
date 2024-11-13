from logging import getLogger

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)

from llama_index.core import VectorStoreIndex

logger = getLogger(__name__)

class ProgressEvent(Event):
    message: str
   
class ChatWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.chat_engine = index.as_chat_engine()
    
    @staticmethod
    def _process_nodes(nodes):
        response = []
        for i in nodes:
            metadata = {k: str(v) for k, v in i.metadata.items()}
            response.append((i.text, metadata, i.node.get_node_info()))
            
        return response

    @step
    async def chat(self, ctx: Context, ev: StartEvent) -> StopEvent:
        message = ev.get('message', None)
        
        if message is None:
            return StopEvent(result=None)
        
        stream_response = await self.chat_engine.astream_chat(message)
        
        async for message in stream_response:
            ctx.write_event_to_stream(ProgressEvent(message=message))

        return StopEvent(result=(stream_response.response, self._process_nodes(stream_response.source_nodes)))


