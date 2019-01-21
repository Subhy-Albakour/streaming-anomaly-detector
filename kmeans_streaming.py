import numpy as numpy

class KMeansStreaming():


    def __init__(self,n_clusters=8, init="k-means++", n_init=10,
                max_iter=300, tol=0.0001, precompute_distances="auto", verbose=0,
                random_state=None, copy_x=True, n_jobs=None, algorithm="auto"):
        # original Kmeans parameters in scikit learn
        self.n_clusters=n_clusters
        self.init=init
        self.n_init=n_init
        self.max_iter=max_iter
        self.tol=tol
        self.precompute_distances=precompute_distances
        self.verbose=verbose
        self.random_state=random_state
        self.copy_x=copy_x
        self.n_jobs=n_jobs
        self.algorithm=algorithm

        ## added 
        self.current_centroids=[]


        def partial_fit(self, X, y=None, classes=None):

            return self


