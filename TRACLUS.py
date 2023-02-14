from trajectory import Trajectory
import numpy as np


class TRACLUS:
    def __init__(self, trajectories):
        # Assert that trajectories is a list of Trajectory objects
        if type(trajectories) != list:
            raise TypeError('Trajectories must be a list of Trajectory objects.')
        elif len(trajectories) == 0:
            raise ValueError('Trajectories must be a list of Trajectory objects.')
        for trajectory in trajectories:
            if type(trajectory) != Trajectory:
                raise TypeError('Trajectories must be a list of Trajectory objects.')

        self.trajectories = trajectories

    def fit(self):
        # Partitioning phase
        partitioned_trajectories = []
        for trajectory in self.trajectories:
            self._partition(trajectory)
            
    def _partition(self, trajectory):
        # Partition a single trajectory
        trajectory.partitions = []
        trajectory.partitions.append(trajectory.points[0]) # Add the starting point
        idx = 0
        while idx + len(trajectory.partitions) < len(trajectory.points):
            continue

