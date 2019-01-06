
import math
import numpy as np

from clustering_feature import ClusteringFeature

class CFTree():

    def __init__(self,T,max_nodes):
        self.root=ClusteringFeature(parent=None,ref=None)
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
        curr_set=self.root.children

        #else :

        while curr_set is not None:


            cf_r=self.nearest(instance,curr_set)

             #----------------------------------------------------------- Remember to correct
            #curr_max_radius=(self.T/(2**(curr_level+3)))**(1/2)
            curr_max_radius=2**((curr_level+3)/2)

            # if the instance if far from all the other CFs in the current set of CFs
            if len(curr_set)==0 or np.linalg.norm(cf_r.ref-instance)>curr_max_radius :
                #
                # creating a new cf and adding it to the first level
                new_cf=ClusteringFeature(parent=cf_f,ref=instance)
                #new_cf.insert(instance)
                cf_f.children.append(new_cf)
                break
            
            # if the instance is not that far
            else:
                # if the current Clustering Feature is not full
                
                if cf_r.cost(instance)<self.T:
                    # print("------------")
                    # print(self.T)
                    # print(cf_r.cost(instance))
                    cf_r.insert(instance)
                    break
                # if the current Clustering Feature is full
                else :
                    curr_set=cf_r.children
                    curr_level=curr_level+1
                    cf_f=cf_r

        if self.root.size() > self.max_nodes:
            self.rebuild(t=2*self.T)
        return self
    

    def rebuild(self,t):
        self.T=t
        s1=[]
        s2=self.root.children

        #----------------------------------------------------------- Remember to correct
        #curr_max_radius=(self.T/16)**(1/2)
        curr_max_radius=4

        while len(s2)>0:
            cf=s2.pop(0)

            nearest=self.nearest(cf.ref,s1)

            if len(s1)==0 or np.linalg.norm(cf.ref-nearest.ref)>curr_max_radius:
                s1.append(cf)

            else:
                if cf.merge_cost(nearest)<=self.T:
                    nearest.merge(cf)
                else:
                    nearest.children.append(cf)
                    cf.parent=nearest

        for cf in s1:
            self.merge_children(cf)
        #self.merge_children(new_root)
        new_root=ClusteringFeature(parent=None,ref=None)
        new_root.children=s1
        self.root=new_root
        
        return self

    
    def merge_children(self,cf):
        
        if len(cf.children)==0:
            return
        i=0
        while i<len(cf.children):
            child=cf.children[i]
            i=i+1
            if cf.merge_cost(child)<=self.T:
                i=i-1
                self.merge_children(child)
                cf.merge(child)
                child=cf.children.pop(i)

        return 
                












    
