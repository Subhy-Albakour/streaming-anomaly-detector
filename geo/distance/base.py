from geo.point import Point
from abc import abstractmethod,ABC

class Distance(ABC):

    @abstractmethod
    def distance(self,p1:Point,p2:Point)->float:
        pass

    @abstractmethod
    def squared_distance(self,p1:Point,p2:Point)->float:
        pass