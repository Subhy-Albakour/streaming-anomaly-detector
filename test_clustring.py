
import numpy as np

# class ClusteringEvaluator:

#     def __init__(self,model,outputfile):
#         self.model=model
#         self.file=outputfile 


#     def partial_fit(self, X, y=None, classes=None):

#         self.model.fit(X)

#         return self

#     def predict(self, X):
#         score=self.model.score(X)
#         print(-score)
#         f=open(self.file,"a")
#         f.write(str(-score)+"\n")
#         f.close()
#         N, D = X.shape
#         return np.zeros(N) 


class ClusteringEvaluator:

    def __init__(self, clustering_model,outputfile,n_update=100):

        self.clustering_model=clustering_model
 
        self.n_update=n_update


        self.batch_counter=0

        self.outputfile= open(outputfile,"w+")
        
        #self.y



    def partial_fit(self, X, y=None, classes=None):

        #------------------------------- detecting pretrain phase--------
        #----------------------TODO (the right way)
        if len(X)>=100:
            #self.clustering_model.partial_fit(X)
            self.clustering_model.fit(X)
            return self


        self.batch_counter += 1
        
        if self.batch_counter<self.n_update:


                self.clustering_model.partial_fit(X)
        else:

                self.clustering_model.fit(X)


        return self

    def predict(self, X):
        score=self.clustering_model.score(X)
        self.outputfile.write(str(-score)+"\n")
        N, D = X.shape
        return np.zeros(N) 
    
class KmeansStreaming:
    
    def __init__(self,kmeans_model):

        self.model=kmeans_model
        self.data=[]
    

    def partial_fit(self,X): # the online operation in streaming configuration

        self.data.extend(X)

        return self

    def fit(self,X): # the offline operation in streaming configuration

        self.data.extend(X)

        self.model.fit(self.data)

        return self

    def transform(self,X): # transform from cluster to distance space
        dist=self.model.transform(X)
        return dist
    
    def score(self,X):
        s=self.model.score(X)
        return s

    
    def predict(self,X):
        y= self.model.predict(X)
        return y


if __name__ == "__main__":

    from sklearn.cluster import KMeans
    from sklearn.cluster import Birch
    from skmultiflow.data import DataStream
    from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    import pandas as pd
    import matplotlib

    from bico import BICO

    def read_data(filename):
        df = pd.read_csv(filename, comment='#')
        y=df['Target'].values
        anoms=(y=="'Anomaly'")
        normal=(y=="'Normal'")
        y[anoms]=1
        y[normal]=0
        X = df.drop(["Target"], axis=1)
        return X,y

    matplotlib.interactive(True)
    data_file="data/mulcross.csv" 
    
    X,y= read_data(data_file)
    stream = DataStream(data=X, y=y) 
    stream.prepare_for_use()

    kmeans1=KMeans(n_clusters=2, random_state=1,init="k-means++")
    kmeans2=KMeans(n_clusters=2, random_state=1,init="k-means++")

    coreset_size=50

    clustering_model1=BICO(kmeans1,max_nodes=coreset_size,thresh=1)
    

    clustering_model2=KmeansStreaming(kmeans2)

    clustering_model3=Birch(n_clusters=2)

    cluster_eval1=ClusteringEvaluator(clustering_model=clustering_model1,outputfile="bico_50.txt")

    #cluster_eval2=ClusteringEvaluator(clustering_model=clustering_model2,outputfile="kmeans_res.txt")

    #cluster_eval3=ClusteringEvaluator(clustering_model=clustering_model3,outputfile="birch.txt")

    h=[cluster_eval1]

    evaluator = EvaluatePrequential(pretrain_size=101, max_samples=50000, show_plot=True, 
                                metrics=['accuracy', 'kappa'], output_file='result.csv', 
                                batch_size=100)


    evaluator.evaluate(stream=stream, model=h)


    