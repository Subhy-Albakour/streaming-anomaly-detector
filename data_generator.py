import numpy as np
from sklearn.datasets import make_blobs

def get_outlier_centers(centers,outlier_distance):
    
    x_centers=[p[0] for p in centers]
    y_centers=[p[1] for p in centers]

    max_x=max(x_centers)
    min_x=min(x_centers)
    max_y=max(y_centers)
    min_y=min(y_centers)
    
    x_range=max_x-min_x
    y_range=max_y-min_y
    
    x1=max_x+outlier_distance*x_range
    x2=min_x-outlier_distance*x_range
    y1=max_y+outlier_distance*y_range
    y2=min_y-outlier_distance*y_range
    outlier_centers=[(x1,y1),(x1,y2),(x2,y1),(x2,y2),(x1,(y1+y2)/2),(x2,(y1+y2)/2),((x1+x2)/2,y1),((x1+x2)/2,y2)]
    
    return outlier_centers
    
def generate_clusters_with_outliers_2d(centers,n_samples=1000,outlier_ratio=0.01,
                                       std=1,outlier_std_ratio=3,outlier_distance=0.5):

    X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=std,
                      centers=centers, shuffle=False, random_state=42)

    n_anoms=int(n_samples*outlier_ratio)
    anoms = np.random.choice(n_samples,n_anoms) 
    outlier_centers=get_outlier_centers(centers,outlier_distance)
    
    X2,y_ = make_blobs(n_samples=n_anoms, n_features=2, cluster_std=std*outlier_std_ratio,
                      centers=outlier_centers, shuffle=False, random_state=42)
    X[anoms]=X2
    y[:]=0
    y[anoms]=1
    
    return X,y
    