import numpy as np
from bico import BICO


class OutlierDetector:

    def __init__(self, clustering_model,thresh=50,n_update=100):

        self.clustering_model=clustering_model
        self.thresh=thresh
        self.n_update=n_update

        self.batch_counter=0


    def partial_fit(self, X, y=None, classes=None):
        

        #------------------------------- detecting pretrain phase--------
        #----------------------TODO (the right way)
        if len(X)>=self.n_update:
            self.clustering_model.online(X)
            self.clustering_model.fit()
            return self


        self.batch_counter += 1

        if self.batch_counter<self.n_update:
            self.clustering_model.online(X)
            return self
        
        self.clustering_model.online(X)
        self.clustering_model.fit()
        return self

    def predict(self, X):
        y=self.clustering_model.predict(X)
        dist=self.clustering_model.transform(X)
        min_dist=dist[range(len(dist)),y]
        logic_classification=(min_dist>=self.thresh)
        outlier_pred=np.array([1 if p else 0 for p in logic_classification ])

        return outlier_pred