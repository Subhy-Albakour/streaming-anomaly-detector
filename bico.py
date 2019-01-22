from cf_tree import CFTree 
from clustering_feature import ClusteringFeature
from geo.point import Point
from streaming_clustering_model import StreamingClusteringModel


class BICO(StreamingClusteringModel):
    
    def __init__(self,kmeans_model,max_nodes=10,dim=2,thresh=0.001):
        # self.max_nodes=max_nodes
        # self.dim=dim
        # self.thresh=thresh
        self.model=kmeans_model
        self.cf_tree=CFTree(dim=2,thresh=0.01,max_nodes= max_nodes)
    

    def partial_fit(self,X): # the online operation in streaming configuration

        self.cf_tree.bulk_insert(X)

        return self

    def fit(self,X): # the offline operation in streaming configuration

        self.cf_tree.bulk_insert(X)

        coreset_centers,weights=self.cf_tree.get_coreset()

        self.model.fit(coreset_centers,sample_weight=weights)

        return self

    def transform(self,X): # transform from cluster to distance space
        dist=self.model.transform(X)
        return dist

    
    def predict(self,X):
        y= self.model.predict(X)
        return y

