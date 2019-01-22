from streaming_clustering_model import StreamingClusteringModel

class BirchStreaming(StreamingClusteringModel):

    def __init__(self,model):
        self.model=model


    def online(self,X):
        

