import math 
import numpy as np


def cal_hull_centroid_percent(model: np.ndarray) -> np.ndarray:
    """
    计算模型质心相对于自身的坐标
    :param model:
    :return: [x%, y%, z%]
    """
    point_list = np.argwhere(model > 0)
    centroid = np.sum(point_list, axis=0) / len(point_list)

    p_min = np.min(point_list, axis=0)
    p_max = np.max(point_list, axis=0)
    percent = (centroid - p_min + 1) / (p_max - p_min + 1)

    return percent, p_min, p_max

def cal_cavity_centroid_percent(model, p_min, p_max) -> np.ndarray:
    point_list = np.argwhere(model > 0)
    centroid = np.sum(point_list, axis=0) / len(point_list)

    percent = (centroid - p_min + 1) / (p_max - p_min + 1)

    return percent
    

def cal_volume(model: np.array) -> int:
    points = np.argwhere(model > 0)

    return len(points)


