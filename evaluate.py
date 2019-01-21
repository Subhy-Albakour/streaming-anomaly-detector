import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data import DataStream

from bico import BICO
from outlier_detector import OutlierDetector
import data_generator


matplotlib.interactive(True)


# 1. Create a stream
centers=[(0,0),(5,5),(-5,7)]
n_samples=1500
random_state=3
X,y=data_generator.generate_clusters_with_outliers_2d(centers,n_samples,outlier_ratio=0.1)


stream = DataStream(data=X, y=y) 
# 2. Prepare for use
stream.prepare_for_use()
# 2. Instantiate the HoeffdingTree classifier
kmeans=KMeans(n_clusters=len(centers), random_state=random_state,init="k-means++")
coreset_size=50
clustering_model=BICO(kmeans,max_nodes=coreset_size)
detector=OutlierDetector(clustering_model,thresh=50,n_update=100)
h = [detector]

# 3. Setup the evaluator

evaluator = EvaluatePrequential(pretrain_size=101, max_samples=1500, show_plot=True, 
                                metrics=['accuracy', 'kappa'], output_file='result.csv', 
                                batch_size=1)

#visualize data

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
# 4. Run
evaluator.evaluate(stream=stream, model=h)



