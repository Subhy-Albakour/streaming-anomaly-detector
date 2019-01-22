import numpy as np
from sklearn.datasets import make_blobs

def get_outlier_centers(cluster_centers,n_outlier_centers=8,outlier_distance=1):
    
    x_centers=[p[0] for p in cluster_centers]
    y_centers=[p[1] for p in cluster_centers]

    max_x=max(x_centers)
    min_x=min(x_centers)
    max_y=max(y_centers)
    min_y=min(y_centers)

    x_origin=(max_x+min_x)/2.
    y_origin=(max_y+min_y)/2.
    
    
    x_range=max_x-min_x
    y_range=max_y-min_y

    radius=((x_range**2+y_range**2)**(1/2))/2.
    #scale the distance of the outliers from the data
    radius=radius*(1+outlier_distance)

    thetas=np.linspace(start=0, stop=2*np.pi, num=n_outlier_centers)
    outlier_centers=[(x_origin+radius*np.sin(theta),y_origin+radius*np.cos(theta)) for theta in thetas]
    
    #x1=max_x+outlier_distance*x_range
    #x2=min_x-outlier_distance*x_range
    #y1=max_y+outlier_distance*y_range
    #y2=min_y-outlier_distance*y_range
    #outlier_centers=[(x1,y1),(x1,y2),(x2,y1),(x2,y2),(x1,(y1+y2)/2),(x2,(y1+y2)/2),((x1+x2)/2,y1),((x1+x2)/2,y2)]
    
    return outlier_centers
    
def generate_clusters_with_outliers_2d(centers,n_samples=1000,outlier_ratio=0.1,outlier_spread=6,
                                       std=1,outlier_std_ratio=3,outlier_distance=0.5):

    X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=std,
                      centers=centers, shuffle=False, random_state=42)

    n_anoms=int(n_samples*outlier_ratio)
    anoms = np.random.choice(n_samples,n_anoms) 

    outlier_centers=get_outlier_centers(centers,n_outlier_centers=outlier_spread,outlier_distance=outlier_distance)
    
    X2,y_ = make_blobs(n_samples=n_anoms, n_features=2, cluster_std=std*outlier_std_ratio,
                      centers=outlier_centers, shuffle=False, random_state=42)
    X[anoms]=X2
    y[:]=0
    y[anoms]=1
    
    return X,y



# ------------ example of usage 

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    centers=[(0,0),(5,5),(-5,7)]
    n_samples=1000
    random_state=3
    X,y=generate_clusters_with_outliers_2d(centers,n_samples,outlier_ratio=0.1,outlier_spread=30,
                                                        outlier_distance=1.5,outlier_std_ratio=3)

    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()