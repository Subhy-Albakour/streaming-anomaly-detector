import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import Birch
from sklearn.cluster import KMeans, MiniBatchKMeans
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data import DataStream

from bico import BICO
from outlier_detector import OutlierDetector,ConstantClassifier
import data_generator


matplotlib.interactive(True)

def generate_data(n_samples):

# 1. Create a stream
    centers=[(0,0),(5,5),(-5,7)]
    X,y=data_generator.generate_clusters_with_outliers_2d(centers,n_samples,outlier_ratio=0.3,outlier_spread=30,
                                                        outlier_distance=1,outlier_std_ratio=2)
    return X,y

def read_data(filename):
    df = pd.read_csv(filename, comment='#')
    y=df['Target'].values
    anoms=(y=="'Anomaly'")
    normal=(y=="'Normal'")
    y[anoms]=1
    y[normal]=0
    X = df.drop(["Target"], axis=1)
    return X,y

n_samples=5000

X,y=generate_data(n_samples)
# data_file="data/mulcross.csv"
# X,y=read_data(data_file)
stream = DataStream(data=X, y=y) 
n_clusters=3
random_state=3


# 2. Prepare for use
stream.prepare_for_use()

kmeans1=KMeans(n_clusters=n_clusters, random_state=random_state,init="k-means++")
kmeans2=KMeans(n_clusters=n_clusters, random_state=random_state,init="k-means++")

coreset_size=100

clustering_model1=BICO(kmeans1,max_nodes=coreset_size,thresh=0.01)
clustering_model2=BICO(kmeans2,max_nodes=coreset_size,thresh=0.01)
clustering_model3=Birch(n_clusters=n_clusters)
clustering_model4=MiniBatchKMeans(n_clusters=n_clusters,random_state=random_state,batch_size=10)
clustering_model5=MiniBatchKMeans(n_clusters=n_clusters,random_state=random_state,batch_size=10)

thresh=3

detector1=OutlierDetector(clustering_model1,thresh=thresh,n_update=10)
detector2=OutlierDetector(clustering_model2,thresh=thresh,n_update=10,remove_outliers=True)
detector3=OutlierDetector(clustering_model3,thresh=thresh,n_update=10)
detector4=OutlierDetector(clustering_model4,thresh=thresh,n_update=10)
detector5=OutlierDetector(clustering_model5,thresh=thresh,n_update=10,remove_outliers=True)

h = [detector1,detector2,detector3,detector4,detector5]

# 3. Setup the evaluator

evaluator = EvaluatePrequential(pretrain_size=101, max_samples=n_samples, show_plot=True, 
                                metrics=['accuracy', 'kappa'], output_file='result.csv', 
                                batch_size=10)

#visualize data

plt.scatter(X[:,0],X[:,1],c=y)

plt.show()
# 4. Run
evaluator.evaluate(stream=stream, model=h)




