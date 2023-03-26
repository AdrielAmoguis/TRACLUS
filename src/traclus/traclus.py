"""
    TRACLUS: A Trajectory Clustering Algorithm (A Partition and Group Framework)
    Implemented for Python 3

    This is an implementation of the TRACLUS algorithm as described in the paper:
    "Trajectory Clustering: A Partition-and-Group Framework"
     by Lee, Han, & Whang (2007) [http://hanj.cs.illinois.edu/pdf/sigmod07_jglee.pdf]

    Implementation Author: Adriel Isaiah V. Amoguis (De La Salle University)
    Implementation Date: 2023-03-19

    This implementation was done as part of the algorithms required for the implementation author's
    undergraduate thesis. The implementation is not guaranteed to be bug-free and may not be optimized
    for certain use-cases. The implementation author is not responsible for any damages caused by the
    use of this implementation. Use at your own risk. End-users are encouraged to examine the code
    in the case of any issues. If you find any bugs, please report them to the implementation author
    via the repository's issues page on GitHub.
"""

import argparse
import numpy as np
from sklearn.cluster import OPTICS
from scipy.spatial.distance import euclidean as d_euclidean

import pickle
import os
import warnings

# UTILITY FUNCTIONS

def load_trajectories(filepath):
    """
        Load the trajectories from a pickle file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError("File not found at {}".format(filepath))

    with open(filepath, 'rb') as f:
        trajectories = pickle.load(f)

    return trajectories

def save_results(trajectories, partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories, filepath):
    """
        Save the results to a pickle file.
    """
    results = {
        'trajectories': trajectories,
        'partitions': partitions,
        'segments': segments,
        'dist_matrix': dist_matrix,
        'clusters': clusters,
        'cluster_assignments': cluster_assignments,
        'representative_trajectories': representative_trajectories
    }
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

def sub_sample_trajectory(trajectory, sample_n=30):
    """
        Sub sample a trajectory to a given number of points.
    """
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be of type np.ndarray")
    elif trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be of shape (n, 2)")

    include = np.linspace(0, trajectory.shape[0]-1, sample_n, dtype=np.int32)
    return trajectory[include]

def calculate_line_euclidean_length(line):
    """
        Calculate the euclidean length of a all points in the line.
    """
    total_length = 0
    for i in range(0, line.shape[0]):
        if i == 0:
            continue
        total_length += d_euclidean(line[i-1], line[i])

    return total_length

def get_point_projection_on_line(point, line):
    """
        Get the projection of a point on a line.
    """

    # Get the slope of the line using the start and end points
    line_slope = (line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) if line[-1, 0] != line[0, 0] else np.inf

    # In case the slope is infinite, we can directly get the projection
    if np.isinf(line_slope):
        return np.array([line[0,0], point[1]])
    
    # Convert the slope to a rotation matrix
    R = slope_to_rotation_matrix(line_slope)

    # Rotate the line and point
    rot_line = np.matmul(line, R.T)
    rot_point = np.matmul(point, R.T)

    # Get the projection
    proj = np.array([rot_point[0], rot_line[0,1]])

    # Undo the rotation for the projection
    R_inverse = np.linalg.inv(R)
    proj = np.matmul(proj, R_inverse.T)

    return proj

def partition2segments(partition):
    """
        Convert a partition to a list of segments.
    """

    if not isinstance(partition, np.ndarray):
        raise TypeError("partition must be of type np.ndarray")
    elif partition.shape[1] != 2:
        raise ValueError("partition must be of shape (n, 2)")
    
    segments = []
    for i in range(partition.shape[0]-1):
        segments.append(np.array([[partition[i, 0], partition[i, 1]], [partition[i+1, 0], partition[i+1, 1]]]))

    return segments

################# EQUATIONS #################

# Euclidean Distance : Accepts two points of type np.ndarray([x,y])
# DEPRECATED IN FAVOR OF THE SCIPY IMPLEMENTATION OF THE EUCLIDEAN DISTANCE
# d_euclidean = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Perpendicular Distance
def d_perpendicular(l1, l2):
    """
        Calculate the perpendicular distance between two lines.
    """
    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    lehmer_1 = d_euclidean(l_shorter[0], ps)
    lehmer_2 = d_euclidean(l_shorter[-1], pe)

    if lehmer_1 == 0 and lehmer_2 == 0:
        return 0
    return (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)#, ps, pe, l_shorter[0], l_shorter[-1]
    
# Parallel Distance
def d_parallel(l1, l2):
    """
        Calculate the parallel distance between two lines.
    """
    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)

    parallel_1 = min(d_euclidean(l_longer[0], ps), d_euclidean(l_longer[-1], ps))
    parallel_2 = min(d_euclidean(l_longer[0], pe), d_euclidean(l_longer[-1], pe))

    return min(parallel_1, parallel_2)

# Angular Distance
def d_angular(l1, l2, directional=True):
    """
        Calculate the angular distance between two lines.
    """

    # Find the shorter line and assign that as l_shorter
    l_shorter = l_longer = None
    l1_len, l2_len = d_euclidean(l1[0], l1[-1]), d_euclidean(l2[0], l2[-1])
    if l1_len < l2_len:
        l_shorter = l1
        l_longer = l2
    else:
        l_shorter = l2
        l_longer = l1

    # Get the minimum intersecting angle between both lines
    shorter_slope = (l_shorter[-1,1] - l_shorter[0,1]) / (l_shorter[-1,0] - l_shorter[0,0]) if l_shorter[-1,0] - l_shorter[0,0] != 0 else np.inf
    longer_slope = (l_longer[-1,1] - l_longer[0,1]) / (l_longer[-1,0] - l_longer[0,0]) if l_longer[-1,0] - l_longer[0,0] != 0 else np.inf

    # The case of a vertical line
    theta = None
    if np.isinf(shorter_slope):
        # Get the angle of the longer line with the x-axis and subtract it from 90 degrees
        tan_theta0 = longer_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    elif np.isinf(longer_slope):
        # Get the angle of the shorter line with the x-axis and subtract it from 90 degrees
        tan_theta0 = shorter_slope
        tan_theta1 = tan_theta0 * -1
        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))
        theta = min(theta0, theta1)
    else:
        tan_theta0 = (shorter_slope - longer_slope) / (1 + shorter_slope * longer_slope)
        tan_theta1 = tan_theta0 * -1

        theta0 = np.abs(np.arctan(tan_theta0))
        theta1 = np.abs(np.arctan(tan_theta1))

        theta = min(theta0, theta1)

    if directional:
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])

    if 0 <= theta < (90 * np.pi / 180):
        return np.sin(theta) * d_euclidean(l_longer[0], l_longer[-1])
    elif (90 * np.pi / 180) <= theta <= np.pi:
        return np.sin(theta)
    else:
        raise ValueError("Theta is not in the range of 0 to 180 degrees.")

# Total Trajectory Distance
def distance(l1, l2, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1):
    """
        Get the total trajectory distance using all three distance formulas.
    """

    perpendicular_distance = d_perpendicular(l1, l2)
    parallel_distance = d_parallel(l1, l2)
    angular_distance = d_angular(l1, l2, directional=directional)

    return (w_perpendicular * perpendicular_distance) + (w_parallel * parallel_distance) + (w_angular * angular_distance)

# Minimum Description Length
def minimum_desription_length(start_idx, curr_idx, trajectory, w_angular=1, w_perpendicular=1, par=True, directional=True):
    """
        Calculate the minimum description length.
    """
    LH = LDH = 0
    for i in range(start_idx, curr_idx-1):
        LH += np.log2(d_euclidean(trajectory[i], trajectory[i+1]))
        if par:
            for j in range(start_idx, i-1):
                # print()
                # print(np.array([trajectory[start_idx], trajectory[i]]))
                # print(np.array([trajectory[j], trajectory[j+1]]))
                LDH += w_perpendicular * d_perpendicular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]))
                LDH += w_angular * d_angular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]), directional=directional)
    if par:
        return LH + LDH
    return LH

# Slope to angle in degrees
def slope_to_angle(slope, degrees=True):
    """
        Convert slope to angle in degrees.
    """
    if not degrees:
        return np.arctan(slope)
    return np.arctan(slope) * 180 / np.pi

# Slope to rotation matrix
def slope_to_rotation_matrix(slope):
    """
        Convert slope to rotation matrix.
    """
    return np.array([[1, slope], [-slope, 1]])

# Get cluster majority line orientation
def get_average_direction_slope(line_list):
    """
        Get the cluster majority line orientation.
        Returns 1 if the lines are mostly vertical, 0 otherwise.
    """
    # Get the average slopes of all the lines
    slopes = []
    for line in line_list:
        slopes.append((line[-1, 1] - line[0, 1]) / (line[-1, 0] - line[0, 0]) if (line[-1, 0] - line[0, 0]) != 0 else 0)
    slopes = np.array(slopes)

    # Get the average slope
    return np.mean(slopes)

# Trajectory Smoothing
def smooth_trajectory(trajectory, window_size=5):
    """
        Smooth a trajectory using a moving average filter.
    """
    # Ensure that the trajectory is a numpy array of shape (n, 2)
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a numpy array")
    elif trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be a numpy array of shape (n, 2)")

    # Ensure that the window size is an odd integer
    if not isinstance(window_size, int):
        raise TypeError("Window size must be an integer")
    elif window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer")

    # Pad the trajectory with the first and last points
    padded_trajectory = np.zeros((trajectory.shape[0] + (window_size - 1), 2))
    padded_trajectory[window_size // 2:window_size // 2 + trajectory.shape[0]] = trajectory
    padded_trajectory[:window_size // 2] = trajectory[0]
    padded_trajectory[-window_size // 2:] = trajectory[-1]

    # Apply the moving average filter
    smoothed_trajectory = np.zeros(trajectory.shape)
    for i in range(trajectory.shape[0]):
        smoothed_trajectory[i] = np.mean(padded_trajectory[i:i + window_size], axis=0)

    return smoothed_trajectory

# Get Distance Matrix
def get_distance_matrix(partitions, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1, progress_bar=False):
    # Create Distance Matrix between all trajectories
    n_partitions = len(partitions)
    dist_matrix = np.zeros((n_partitions, n_partitions))
    for i in range(n_partitions):
        if progress_bar: print(f'Progress: {i+1}/{n_partitions}', end='\r')
        for j in range(i+1):
            dist_matrix[i,j] = dist_matrix[j,i] = distance(partitions[i], partitions[j], directional=directional, w_perpendicular=w_perpendicular, w_parallel=w_parallel, w_angular=w_angular)
            print(f'Progress: {i+1}/{n_partitions}', end='\r')

    # Main Diagonal
    for i in range(n_partitions):
        dist_matrix[i,i] = 0

    # Check for nans and warn if any are found
    if np.isnan(dist_matrix).any():
        warnings.warn("Distance matrix contains NaN values")
    
        # Replace the nans with the maximum value
        dist_matrix[np.isnan(dist_matrix)] = 9999999

    return dist_matrix

#############################################

def partition(trajectory, directional=True, progress_bar=False, w_perpendicular=1, w_angular=1):
    """
        Partition a trajectory into segments.
    """

    # Ensure that the trajectory is a numpy array of shape (n, 2)
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a numpy array")
    elif trajectory.shape[1] != 2:
        raise ValueError("Trajectory must be a numpy array of shape (n, 2)")

    # Initialize the characteristic points, add the first point as a characteristic point
    cp_indices = []
    cp_indices.append(0)

    traj_len = trajectory.shape[0]
    start_idx = 0
    
    length = 1
    while start_idx + length < traj_len:
        if progress_bar:
            print(f'\r{round(((start_idx + length) / traj_len) * 100, 2)}%', end='')
        # print(f'Current Index: {start_idx + length}, Trajectory Length: {traj_len}')
        curr_idx = start_idx + length
        # print(start_idx, curr_idx)
        # print(f"Current Index: {curr_idx}, Current point: {trajectory[curr_idx]}")
        cost_par = minimum_desription_length(start_idx, curr_idx, trajectory, w_angular=w_angular, w_perpendicular=w_perpendicular, directional=directional)
        cost_nopar = minimum_desription_length(start_idx, curr_idx, trajectory, par=False, directional=directional)
        # print(f'Cost with partition: {cost_par}, Cost without partition: {cost_nopar}')
        if cost_par > cost_nopar:
            # print(f"Added characteristic point: {trajectory[curr_idx-1]} with index {curr_idx-1}")
            cp_indices.append(curr_idx-1)
            start_idx = curr_idx-1
            length = 1
        else:
            length += 1
    
    # Add last point to characteristic points
    cp_indices.append(len(trajectory) - 1)
    # print(cp_indices)
    
    return np.array([trajectory[i] for i in cp_indices])

# Get Representative Trajectory
def get_representative_trajectory(lines, min_lines=3):
    """
        Get the sweeping line vector average.
    """
    # Get the average rotation matrix for all the lines
    average_slope = get_average_direction_slope(lines)
    rotation_matrix = slope_to_rotation_matrix(average_slope)

    # Rotate all lines such that they are parallel to the x-axis
    rotated_lines = []
    for line in lines:
        rotated_lines.append(np.matmul(line, rotation_matrix.T))

    # Let starting_and_ending_points be the set of all starting and ending points of the lines
    starting_and_ending_points = []
    for line in rotated_lines:
        starting_and_ending_points.append(line[0])
        starting_and_ending_points.append(line[-1])
    starting_and_ending_points = np.array(starting_and_ending_points)

    # Sort the starting and ending points by their x-coordinate
    starting_and_ending_points = starting_and_ending_points[starting_and_ending_points[:, 0].argsort()]

    # Perform the sweeping line algorithm
    representative_points = []
    for p in starting_and_ending_points:
        # Let num_p be the number of lines that contain the x-value of p
        num_p = 0
        for line in rotated_lines:
            # Sort the line points by their x-coordinate
            point_sorted_line = line[line[:, 0].argsort()]
            # print(line[0, 0], p[0], line[-1, 0])
            if point_sorted_line[0, 0] <= p[0] <= point_sorted_line[-1, 0]:
                num_p += 1

        # If num_p is greater than or equal to min_lines, then add p to representative_points
        if num_p >= min_lines:
            # Compute the average y-value of all lines that contain the x-value of p
            y_avg = 0
            for line in rotated_lines:
                point_sorted_line = line[line[:, 0].argsort()]
                if point_sorted_line[0, 0] <= p[0] <= point_sorted_line[-1, 0]:
                    y_avg += (point_sorted_line[0, 1] + point_sorted_line[-1, 1]) / 2
                    # print((point_sorted_line[0, 1] + point_sorted_line[-1, 1]) / 2)
                    # y_avg += line[np.argmin(np.abs(line[:, 0] - p[0])), 1]
            y_avg /= num_p
            # Add the p and its average y-value to representative_points
            representative_points.append(np.array([p[0], y_avg]))

    # Early return if there are no representative points
    if len(representative_points) == 0:
        warnings.warn("WARNING: No representative points were found.")
        return np.array([])

    # Undo the rotation for the generated representative points
    representative_points = np.array(representative_points)
    representative_points = np.matmul(representative_points, np.linalg.inv(rotation_matrix).T)
    
    return representative_points


def traclus(trajectories, max_eps=None, min_samples=10, directional=True, use_segments=True, clustering_algorithm=OPTICS, mdl_weights=[1,1,1], d_weights=[1,1,1], progress_bar=False):
    """
        Trajectory Clustering Algorithm
    """
    # Ensure that the trajectories are a list of numpy arrays of shape (n, 2)
    if not isinstance(trajectories, list):
        raise TypeError("Trajectories must be a list")
    for trajectory in trajectories:
        if not isinstance(trajectory, np.ndarray):
            raise TypeError("Trajectories must be a list of numpy arrays")
        elif len(trajectory.shape) != 2:
            raise ValueError("Trajectories must be a list of numpy arrays of shape (n, 2)")
        elif trajectory.shape[1] != 2:
            raise ValueError("Trajectories must be a list of numpy arrays of shape (n, 2)")

    # Partition the trajectories
    if progress_bar:
        print("Partitioning trajectories...")
    partitions = []
    i = 0
    for trajectory in trajectories:
        if progress_bar:
            print(f"\rTrajectory {i + 1}/{len(trajectories)}", end='')
            i += 1
        partitions.append(partition(trajectory, directional=directional, progress_bar=False, w_perpendicular=mdl_weights[0], w_angular=mdl_weights[2]))
    if progress_bar:
        print()

    # Get the segments for each partition
    segments = []
    if use_segments:
        if progress_bar:
            print("Converting partitioned trajectories to segments...")
        i = 0
        for parts in partitions:
            if progress_bar:
                print(f"\rPartition {i + 1}/{len(parts)}", end='')
            segments += partition2segments(parts)
    else:
        segments = partitions

    # Get distance matrix
    dist_matrix = get_distance_matrix(segments, directional=directional, w_perpendicular=d_weights[0], w_parallel=d_weights[1], w_angular=d_weights[2], progress_bar=progress_bar)

    # Group the partitions
    if progress_bar:
        print("Grouping partitions...")
    clusters = []
    clustering_model = None
    if max_eps is not None:
        clustering_model = clustering_algorithm(max_eps=max_eps, min_samples=min_samples)
    else:
        clustering_model = clustering_algorithm(min_samples=min_samples)
    cluster_assignments = clustering_model.fit_predict(dist_matrix)
    for c in range(min(cluster_assignments), max(cluster_assignments) + 1):
        clusters.append([segments[i] for i in range(len(segments)) if cluster_assignments[i] == c])

    if progress_bar:
        print()

    # Get the representative trajectories
    if progress_bar:
        print("Getting representative trajectories...")
    representative_trajectories = []
    for cluster in clusters:
        representative_trajectories.append(get_representative_trajectory(cluster))
    if progress_bar:
        print()

    return partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories

# Create the script version that takes in a file path for inputs
if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Trajectory Clustering Algorithm")
    parser.add_argument("input_file", help="The input file path (pickle format)")
    parser.add_argument("output_file", help="The output file path (pickle format)")
    parser.add_argument("-e", "--eps", help="The epsilon value for the clustering algorithm", type=float, default=2)
    parser.add_argument("-m", "--min_samples", help="The minimum samples value for the clustering algorithm", type=int, default=3)
    parser.add_argument("-p", "--progress_bar", help="Show the progress bar", action="store_true")
    args = parser.parse_args()

    # Load the trajectories
    trajectories = load_trajectories(args.input_file)

    # Run the TraClus algorithm
    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = traclus(trajectories, eps=args.eps, min_samples=args.min_samples, progress_bar=args.progress_bar)

    # Save the results
    save_results(args.output_file, trajectories, partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories)