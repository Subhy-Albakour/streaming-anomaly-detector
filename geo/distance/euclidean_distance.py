from base import Distance
from geo.point import Point

class EuclideanDistance(Distance):

    def __init__(self):
        pass

    def distance(self,p1:Point,p2:Point)->float:
        
        d=self.squared_distance(p1,p2)
        return d**(1/2)

    def squared_distance(self,p1:Point,p2:Point)->float:
        d=p1-p2
        return d*d