import numpy as np
from bico import BICO


class OutlierDetector:

    def __init__(self, clustering_model,thresh=50,n_update=100,remove_outliers=False):

        self.clustering_model=clustering_model
        self.thresh=thresh
        self.n_update=n_update
        self.remove_outliers=remove_outliers

        self.batch_counter=0
        
        self.last_prediction=None
        #self.y



    def partial_fit(self, X, y=None, classes=None):
        
        # print("fit-------------")
        # print(y)
        #self.y=y
        #------------------------------- detecting pretrain phase--------
        #----------------------TODO (the right way)
        if len(X)>=100:
            self.clustering_model.online(X)
            self.clustering_model.fit()
            return self


        self.batch_counter += 1
        
        if self.remove_outliers:
            ind=(self.last_prediction==0)# the indecies of non outliers
            # print("outlier removed:------",len(X)-sum(ind))
            # print(self.clustering_model.cf_tree.root.size())
            self.clustering_model.online(X[ind])
        else:
            # print("Deafualt --------")
            # print(self.clustering_model.cf_tree.root.size())
            self.clustering_model.online(X)
        
        if self.batch_counter==self.n_update:
            self.clustering_model.fit()

        # if self.batch_counter<self.n_update:
        #     if self.remove_outliers:
        #         ind=(self.last_prediction==0)# the indecies of non outliers
        #         self.clustering_model.online(X[ind])
        #     else:
        #         self.clustering_model.online(X)
        #     return self
        # if self.remove_outliers:
        #     ind=(self.last_prediction==0)
        #     self.clustering_model.online(X[ind])
        #     self.clustering_model.fit()

        # self.clustering_model.online(X)
        # self.clustering_model.fit()

        return self

    def predict(self, X):
        y=self.clustering_model.predict(X)
        dist=self.clustering_model.transform(X)
        min_dist=dist[range(len(dist)),y]
        logic_classification=(min_dist>=self.thresh)
        outlier_pred=np.array([1 if p else 0 for p in logic_classification ])
        if self.remove_outliers:
            #print("--------- rem  ",self.clustering_model.cf_tree.get_coreset())
            self.last_prediction=outlier_pred
        # else :
        #     print("--------- def  ",self.clustering_model.cf_tree.get_coreset())
        
        # print("outlier-----------------")
        # print(outlier_pred)
        #print(min_dist)
        # print(dist)
        return outlier_pred
    

# class ConstantClassifier:
#     def __init__(self):
#         pass
    
#     def partial_fit(self,X,y=None,classes=None):
        
#         return self
#     def predict(self,X):
#         n,=X.shape
        
#         return np.zeros(n)

class ConstantClassifier:

    def __init__(self):
        pass

    def partial_fit(self, X, y=None, classes=None):

        return self

    def predict(self, X):
        N, D = X.shape
        return np.zeros(N) 