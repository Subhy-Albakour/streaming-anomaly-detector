from cf_tree import CFTree 
from clustering_feature import ClusteringFeature
import numpy as np


## to keep imports uptodata

import clustering_feature
import cf_tree
import imp
imp.reload(clustering_feature)
imp.reload(cf_tree)

data=np.array([[1,0],[1,2],[1,3],[2,1],[1,4],[4,1],[1,0],[5,66]])

tree=CFTree(T=10,max_nodes= 10)
for i in range(len(data)):
    tree.insert(data[i])
print(tree.root)



tree.rebuild(25)

print(tree.root)
