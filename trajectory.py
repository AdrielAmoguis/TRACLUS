import numpy as np

class Trajectory:
    def __init__(self, points):
        if type(points) != 'numpy.ndarray':
            raise TypeError('Points must be a 2-dimensional numpy array containing all the trajectory points.')
        elif points.shape[1] != 2:
            raise ValueError('Points must be a 2-dimensional numpy array containing all the trajectory points.')
            
        self.points = points
        self.partitions = []