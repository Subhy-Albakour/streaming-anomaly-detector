import numpy as np

class ClusteringFeature():

    """ ClsteringFeature
    
    This calss represents the elements of the summary of a data set
    it compactes a subset S of the original data set.
    it stores the number of elements |S|, the sum of these elements sum(S), the sum of the squares sum(S^2)
    
    Parameters
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


    def __init__(self,parent):
        self.num=0
        self.s=0
        self.ss=0
        self.ref=None 
        self.children=[]
        self.parent=parent
        
    
    def size():
        if self.children is None:
            return 0
       
        s=1
        for cf in slef.children:
            s=s+cf.size()
        return s
        
    

    def insert(instance):
        """ Inserts a  new data instance to this Clustering Feature (CF)
        
        Parameters
        ----------
        instance: Numpy.ndarray of shape (d, )
            a single instance of the data (for R^d)

        Retruns
        -------
        self

        """
        self.num=self.num+1
        self.s=self.s+instance
        self.ss=self.ss+sum(instance**2)

        return self
    
    def cost(instance=None):
        """ cost
        
        Computes the cost of this ClusteringFeature (the sum of distances to the centroid)

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
        if instance in not None:
            res=self.ss - (1.0/self.num) * sum(self.s**2)
        else:
            num_new=self.num+1
            s_new=self.s+instance
            ss_new=self.ss+sum(instance**2)
            res=ss_new-(1.0/num_new) * sum(s_new * s_new)

        return res




    
    
    # def distance(self,instance):
    #     """ distance
        
    #     Computes the distance between this ClusteringFeature and instance

    #     Parameters
    #     ----------
    #     instance: Numpy.ndarray of shape (d, )
    #         a data instance to calculate the distance between 'as defined in the paper'
        
    #     Returns
    #     -------
    #     float 
    #         the distance between the instance and the clustering feature

    #     """
    #     var=self.ss - (1.0/self.num) * sum(self.s*self.s)
    #     num_new=self,num+1
    #     s_new=self.s+instance
    #     ss_new=self.ss+sum(instance*instance)
    #     var_new=ss_new-(1.0/num_new) * sum(s_new * s_new)

    #     return var_new - var
