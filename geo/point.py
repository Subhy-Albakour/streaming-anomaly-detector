import numpy as np


class Point:
    """
    Point class for geometry representation in bico data structures
    """

    def __init__(self, point: np.ndarray):
        """
        :param point:
            1-D Numpy array representing a geometry
        """
        self.p =np.array(point) # define an array in case of passing a list instead of an ndarray

    def set_point(self, point: np.ndarray):
        self.p = point

    def __add__(self, other: 'Point') -> 'Point':

        return Point(self.p + other.p)
    
    def __sub__(self, other: 'Point') -> 'Point':

        return Point(self.p - other.p)

    def __iadd__(self, other: 'Point') -> 'Point':
        self.p += other.p
        return self

    def __mul__(self, other: 'Point') -> float:
        """
        Inner product of two points
        :param other:
            Second geometry for inner product computation
        :return:
            Result of inner product as a float.
        """
        return np.inner(self.p, other.p)
    def __pow__ (self,n):

        return Point(self.p**n)

    def scalar_mul(self, scalar: float) -> 'Point':
        """
        Scalar multiplication of this geometry with a scalar.
        :param scalar:
            Scalar as a float
        :return:
            New geometry with the result of the scalar multiplication
        """
        return Point(scalar * self.p)


    def get_dim(self):
        return len(self.p)

    def __str__(self) -> str:
        return str(self.p)

class CorePoint():

    def __init__(self,point,weight):
        self.center=point
        self.weight=weight

    def __str__(self):
        return "("+str(self.center)+","+str(self.weight)+")"