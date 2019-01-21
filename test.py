from cf_tree import CFTree 
from clustering_feature import ClusteringFeature
import numpy as np
import matplotlib.pyplot as plt


## to keep imports uptodata

import clustering_feature
import cf_tree
import imp
imp.reload(clustering_feature)
imp.reload(cf_tree)
from geo.point import Point

data=np.array([[1,0],[1,2],[1,3],[2,1],[1,4],[4,1],[1,0],[5,66]])

tree=CFTree(dim=2,thresh=20,max_nodes= 10)
for i in range(len(data)):
    p=Point(data[i])
    tree.insert(p)
print(tree.root)

tree.rebuild(100)
print(tree.root)

coreset,ws=tree.get_coreset()
for c in coreset:
    print(c)

# xs=[c.center.p[0] for c in coreset]
# ys=[c.center.p[1] for c in coreset]
# #ws=[50*c.weight for c in coreset]

plt.subplot(121)
plt.scatter(coreset[:,0],coreset[:,1],s=ws)
plt.title("Coreset")
plt.show()
from sklearn.cluster import KMeans
k=Kmeans(init=)
np.arange()