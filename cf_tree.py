
import math
import numpy as np

class CFTree():

    def __init__(self, root,T,max_nodes):
        self.root=root
        self.T=T
        self.max_nodes=max_nodes
    

    def nearest(self,instance,cf_set):
        min=math.inf
        res=None
        for cf in cf_set:
            dis=np.linalg.norm(instance-cf.ref)
            if dis < min :
                min=dis
                res=cf
        return res
    


    def insert(self,instance):
        cf_f=self.root
        curr_level=1
        curr_set=root.children
       



        while curr_set is not None:

            cf_r=nearest(instance,curr_set)
            curr_max_radius=2**((curr_level+3)/2)

            dis=np.linalg.norm(cf_r.ref-instance)
            # if the instance if far from all the other CFs in the current set of CFs
            if dis>curr_max_radius :
                #
                # creating a new cf and adding it to the first level
                new_cf=ClusteringFeature(parent=root)
                new_cf.insert(instance)
                cf_f.children.append(new_cf)
            
            # if the instance is not that far
            else:
                # if the current Clustering Feature is not full
                if cf_r.cost(instance)<self.T:
                    cf_r.insert(instance)
                    break
                # if the current Clustering Feature is full
                else :
                    curr_set=cf_r.children
                    curr_level=curr_level+1
                    cf_f=cf_r

        if slef.root.size > self.max_nodes:
            rebuild(t=2*T)
    
    
    def rebuild(self,t):
        self.T=t







    
