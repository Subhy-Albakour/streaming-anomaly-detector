"""

This module is responsible for the logic of Clustering Features operations

"""

import numpy as np

from geo.point import Point,CorePoint



class ClusteringFeature():

    """ ClsteringFeature
    
    This calss represents the elements of the summary of a data set
    it compactes a subset S of the original data set.
    it stores the number of elements |S|, the sum of these elements sum(S), the sum of the squares sum(S^2).

    This class represnts the nodes of CF tree.

    
    Traits
    ----------
    num: int 
        the number of the original data instances S that are represented by this same Clustering Feature object.

    s: Numpy.ndarray of shape (d,)
        the sum of the original data instances.

    ss: float
        the sum of the square of the Euclidian lengths of the original instances.
    
    ref: Numpy.ndarray of shape (d, )
        the reference instance (point) of this ClusteringFeature

    children: list of ClusteringFeatrue  
        the children of this ClusteringFeature object as a node of ClusterinFeatureTree 

    """


    def __init__(self, parent: 'ClusteringFeature', ref:Point):
        """ Creates new ClusteringFeature (CF).
        
        Parameters
        ----------
        parent: ClusteringFeature
            the parent node of this Clustering Feature 
        
        ref: Numpy.ndarray of shape (d, )
            the reference instance of this Clustering Feature 

        """

        # The total number of instances that this CF represents (summarizes)
        self.num=0

        # The linear sum of all instances that this CF represents
        self.s=None

        # The sum of of all Euclidean norms of all instances that this CF represnts
        self.ss=0

        # The reference instance  (point) of this CF
        self.ref=ref

        # List of all children of this CF in CFtree 
        self.children=[]

        # The parent node of this Clustring Feature Node
        self.parent=parent

        # The reference instance should be inserted into this Clustering Feature. 
        if ref is not None:
            self.insert(ref)
        
        return 
    
    

    def __str__(self):
        """ Creates a string represenation of this Clustering Feature """

        return self.to_string("")

        
    def to_string(self, pref:str):
        """ Recursively builds the string represntation of this Clustering Feature
        
        Parameters
        ----------
        pref: string
            the prefix that should be added before the representation for it to look like a tree
         
        """

        # One line represenation of this Clustring Feature 
        s="|num = "+str(self.num) +", ss=  "+str(self.ss)+", s= "+str(self.s)

        # the prepresnataion of the children 
        ch="\n".join([child.to_string(pref+"   ") for child in self.children])

        return pref+ s+"\n"+ch


    def size(self):
        """ Recursively computes the size of the tree rooted at this Clustering Feature"""

        if self.children is None:
            return 0
       
        s=1
        for cf in self.children:
            s=s+cf.size()
        return s

    

    def insert(self,point:Point):
        """ Inserts a  new data instance to this Clustering Feature (CF)
        
        Parameters
        ----------
        instance: Numpy.ndarray of shape (d, )
            a single instance of the data (from the space R^d)


        """

        self.num=self.num+1
        if self.s is None:
            self.s=point
        else:
            self.s += point
        self.ss=self.ss+sum((point**2).p)

        return 

    
    def cost(self, point=None):
        """ cost
        
        Computes the cost of this ClusteringFeature (the sum of squared distances to the centroid)

        Parameters
        ----------
        instance: Numpy.ndarray of shape (d, ), optional (default= None)
            a data instance to compute the new cost supposing that this instance is added to this ClusteringFeatrue
        
        Returns
        -------
        float 
            the distance between the instance and the clustering feature

        """

        res=0
        # compute the cost of this CF
        if point is  None:
            res=self.__cf_cost(self.num,self.s,self.ss)

        # compute the cost of the new CF supposing that 'insatnce' is inserted
        else:
            num_new=self.num+1
            s_new=self.s+point
            ss_new=self.ss+sum((point**2).p)
            res=self.__cf_cost(num_new,s_new,ss_new)

        return res

    def merge(self,other_cf:'ClusteringFeature'):
        """ Merges this Clustring Feature with another Clustering Feature

        Parameters
        ----------
        other_cf: ClusteringFeature
            The other Clustering Feature object to be merged


        """

        # Updating the Clustering Features
        self.num=self.num+other_cf.num
        self.s=self.s+other_cf.s
        self.ss=self.ss+other_cf.ss

        # Making the children of the other object children of the new merged  object
        for child in other_cf.children:
            self.children.append(child)
            child.parent=self
        
        return 


    
    def merge_cost(self,other_cf:'ClusteringFeature'):
        """ Computes the cost of the merged object but without actually performing the merge
        
        Notes
        _____
        This function does not perform the merge, it just computes what the cost would be in case of merging
        
        Parameters
        ----------
        other_cf: Clustering Feature
            The other Clustering Feature object to be (hypothetically)merged 
        
        Returns
        -------
        float
            The cost of the Clustring Feature resulted from the merge 


        """
        n=self.num+other_cf.num
        s=self.s+other_cf.s
        ss=self.ss+other_cf.ss

        return self.__cf_cost(n,s,ss)
    

    def get_center(self)->Point:
        if self.s is None:
            return None
        center=CorePoint(self.s.scalar_mul(1/self.num),self.num)
        return center
    

    def get_descendent_centers(self):
        if self.children is None:
            return []
        
        res=[]
        center=self.get_center()
        if  center is not None :
            res=[center]
        for cf in self.children:
            res.extend(cf.get_descendent_centers())
        return res
        

    def __cf_cost(self, n, s:Point, ss):
        """ Computes the cost of a cluster using its features

        Parameters
        ----------
        n: int 
            the number of the original data instances

        s: Numpy.ndarray of shape (d,)
            the sum of the original data instances.

        ss: float
            the sum of the square of the Euclidian lengths of the original instances.
        """

        return ss - (1.0/n) * sum((s**2).p)
    

