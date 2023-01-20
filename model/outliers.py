import math
from scipy.stats import mode
import numpy as np

#outlier detection
def outliers_(euclidean_dist):

    euclidean_dist_mode = mode(euclidean_dist)[0][0]
    euclidean_dist_ten_percent = round(max(euclidean_dist)/9)

    lower_bound = euclidean_dist_mode - euclidean_dist_ten_percent
    upper_bound = euclidean_dist_mode + euclidean_dist_ten_percent

    outliers = [x for x in euclidean_dist if x <= lower_bound or x >= upper_bound]


    return outliers
def dist(points):
    euclidean_dist = []
    for i in range(0, len(points), 2):
        try:
            euclidean_dist.append(math.dist(points[i],points[i+1]))
        except:
            continue
    return euclidean_dist
def get_index(list,outliers):
    index = []
    for idx, _ in enumerate(list):
        if list[idx] in outliers:
            index.append(idx*2)
    return index
def filtered_points(points):
    list_dist = dist(points)
    outliers = outliers_(list_dist)
    index = get_index(list_dist,outliers)
    return np.asarray([i for j, i in enumerate(points) if j not in index])
def get_corners(points):
    min_idx = np.argmin(points,axis=0)
    max_idx = np.argmax(points,axis=0)
    min_idx_axis1 = np.argmin(points,axis=1)
    bottom_left,top_left = points[min_idx]
    bottom_right,top_right = points[max_idx]

    return bottom_left,top_left,bottom_right,top_right,min_idx_axis1