"""
This module contains the logic to build the Clustring Feature Tree and keep it valid (according to CF tree conditions)

"""

import math
import numpy as np
from geo.point import Point

# Local import
from clustering_feature import ClusteringFeature

class CFTree():
    """ This class represents a tree of Clustering Featrues 
    
    This calss encapsulates all the logic of creating a Clustring Featrue and keep it in the right form
    and that is by reconstructing it when it violates the Clustering Feature tree. 
    
    """

    def __init__(self,dim,thresh,max_nodes):
        """ Creates a new Clustring Featrue Tree object

        Parameters
        ----------
        thresh: float
            The cost threshold that a clustring feature node should not exceed.

        max_nodes: int
            The maximum number of nodes that should the tree contain.
        """

        # The dimension of the input data space
        self.dim=dim

        # Creating a fictional root
        self.root=ClusteringFeature(parent=None,ref=None)

        # The cost threshold
        self.thresh=thresh

        # The maximum number of nodes that the tree should contain
        self.max_nodes=max_nodes
    

    def nearest(self,point:Point,cf_set):
        """ Finds the nearest Clustring Feature to an instance form a set Clustring Features 

        Paramters
        ---------
        instance: Numpy.ndarray of shape (d, )
            The instance to find the nearest CF to.
        
        cf_set: Iterable of ClusteringFeature
            The set of considered CFs to search in
        
        Returns
        -------
        ClusteringFeature
            the nearest ClustringFeature 'of cf_set' to 'instance'
        
        """

        #------------------------------------------------------------to be improved-----------------------------------
        import math
        min=math.inf

        res=None
        
        for cf in cf_set:
            dis=np.linalg.norm(point.p-cf.ref.p)
            if dis < min :
                min=dis
                res=cf
        
        return res
    


    def insert(self,point:Point):
        """ Inserts a  new data instance to this CFTree object
        
        Parameters
        ----------
        instance: Numpy.ndarray of shape (d, )
            a single instance of the data (of the space R^d)


        """

        cf_f=self.root
        curr_level=1
        curr_set=self.root.children



        while curr_set is not None:


            cf_r=self.nearest(point,curr_set)

            #----------------------------------------------------------- to Correct (uncomment the next line and comment the one afterwards)------------------------------------------
            #curr_max_radius=(self.T/(2**(curr_level+3)))**(1/2)
            #curr_max_radius=2**((curr_level+3)/2)
            curr_max_radius=self.get_radius(curr_level)

            # if the instance is far from all the other CFs in the current set of CFs
            if len(curr_set)==0 or np.linalg.norm(cf_r.ref.p-point.p)>curr_max_radius :

                # creating a new cf and adding it to the first level
                new_cf=ClusteringFeature(parent=cf_f,ref=point)

                cf_f.children.append(new_cf)
                break
            
            # if the instance is not that far
            else:
                # if the current Clustering Feature is not full
                
                if cf_r.cost(point)<self.thresh:

                    cf_r.insert(point)
                    break

                # if the current Clustering Feature is full
                else :
                    curr_set=cf_r.children
                    curr_level=curr_level+1
                    cf_f=cf_r

        # Rconstructing the tree when if violates the size condition
        if self.root.size() > self.max_nodes:
            self.rebuild(thresh=2*self.thresh)
        
        return 

    def bulk_insert(self,points): # points : np.ndarray of the shape (:,dim)
        for point in points:
            self.insert(Point(point)) 
        return
    

    def rebuild(self,thresh):
        """ Rebuilds the tree using a new cost threshold

        Parameters
        ----------
        thresh: float
            The new cost threshold

        """
        self.thresh=thresh
        # The new first level nodes
        s1=[]

        # the old first level nodes
        s2=self.root.children

        #----------------------------------------------------------- Remember to correct
        #curr_max_radius=(self.T/16)**(1/2)
        #curr_max_radius=4
        curr_max_radius=self.get_radius(level=1)
        # While there is more nodes in the first level of the old tree
        while len(s2)>0:

            cf=s2.pop(0)

  
            nearest=self.nearest(cf.ref,s1)


            if len(s1)==0 or np.linalg.norm(cf.ref.p-nearest.ref.p)>curr_max_radius:

                # move cf one level up
                s1.append(cf)

            else:
                
                if cf.merge_cost(nearest)<=self.thresh:

                    # Merging cf with the nearest CF in the first level
                    nearest.merge(cf)
                else:

                    # Adding cf to the second level on the new tree
                    nearest.children.append(cf)
                    cf.parent=nearest

        # Mreging each node with its parent if possible
        for cf in s1:
            self.merge_children(cf)

        # Attaching the new tree to a new fictional root
        new_root=ClusteringFeature(parent=None,ref=None)
        new_root.children=s1
        self.root=new_root
        
        return 

    
    def merge_children(self,cf):
        """ Traverses the tree rooted at a given node bottom up and megres each node to its parent if possible

        Parameters
        ----------
        cf: ClsteringFeature
            The root node 
        """
        
        if len(cf.children)==0:
            return
        i=0
        while i<len(cf.children):
            child=cf.children[i]
            i=i+1
            if cf.merge_cost(child)<=self.thresh:
                i=i-1

                # Recuresive call on the children before the parent (bottm up traverse)
                self.merge_children(child)
                cf.merge(child)

                # Removeing the child after merging 
                child=cf.children.pop(i)

        return 

    def get_thresh(self):
        return self.thresh
    
    def get_radius(self,level:int)->float:
        
        return (self.thresh/(1<<(level+3)))#**(1/2)

    def get_coreset(self):
        centers=self.root.get_descendent_centers()
        weights=[c.weight for c in centers]
        centers=np.vstack([c.center.p for c in centers])
        return (centers,weights)

        