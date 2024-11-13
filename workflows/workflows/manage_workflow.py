import os
from logging import getLogger

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step
)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

logger = getLogger(__name__)
   
class ManageWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index

    @step
    async def manage(self, ev: StartEvent) -> StopEvent:
        type = ev.get('type', None)
        file_path = ev.get('file_path', None)
        
        if type is None or file_path is None:
            return StopEvent(result=None)
        
        if type == 'add':
            delete = False
            if file_path is None:
                bytestream = ev.get('bytestream', None)
                file_name = ev.get('file_name', None)
                
                if bytestream is None or file_name is None:
                    return StopEvent(result=None)
                
                file_path = f'/tmp/{os.path.basename(file_name)}'
                    
                with open(file_path, 'wb') as f:
                    f.write(bytestream)      
                    
                delete = True
            
            nodes = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            for transform in Settings.transformations:
                nodes = transform(nodes)
                
            await self.index.insert_nodes(nodes)
            
            if delete:
                os.remove(file_path)
            
            return StopEvent(result=list([node.id_ for node in nodes]))
        elif type == 'remove':
            ids = ev.get('ids', None)
            
            if ids is None:
                return StopEvent(result=None)
            
            self.index.delete_nodes(ids)
            return StopEvent(result=True)
        
        return StopEvent(result=None)



