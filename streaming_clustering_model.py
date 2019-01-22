
from abc import abstractclassmethod,ABC

class StreamingClusteringModel(ABC):


    def __init__(self):
        pass
    

    @abstractclassmethod
    def partial_fit(slef,X):
        """
        the online operation, it has to be performed over each batch 
        """
        pass
    
    @abstractclassmethod
    def fit(self,X):
        """
        the offline operation,
        it is perforemed each  number of batches
        """
        pass
    
    @abstractclassmethod
    def transform(self,X):
        """
        computes the distances of the data sample to the clusters
        """
        pass

    @abstractclassmethod
    def predict(self,X):
        """
        assigns data to the clusters
        """
        pass
