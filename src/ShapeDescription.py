import os
import csv
import numpy as np
import math

from voxel_array import get_6_neighbor_surface

test_result_path = './data/surface_voxel'

def sample_n2_distance(model: np.ndarray, name: str, sample_size = 512, neighbor = 6):
    """
    计算模型表面点之间的距离分布，并返回距离列表和最大距离。
    
    Args:
        model (np.ndarray): 输入的三维体素模型。
        sample_size (int, optional): 要采样的表面点数量。默认为 512。
        neighbor (int, optional): 邻域类型，可以是6或26。默认为6。
    
    Returns:
        (list, float): 返回距离列表和模型对角线的最大距离。
    """
    if neighbor == 6:
        surface_voxel = get_6_neighbor_surface(model)
    distance_list = []
    # 找到表面体素的位置
    surface = np.argwhere(surface_voxel == 2)

    # Test
    filename = f'{name}.csv'
    with open(os.path.join(test_result_path, filename), 'w', newline='') as file:
        writer = csv.writer(file)
        for item in list(surface):
            writer.writerow([item])

    # 如果表面体素数量小于等于采样数量，使用所有表面体素
    if len(surface) <= sample_size:
        p = surface
    else:
        # 否则，从表面体素中均匀采样指定数量的点
        p = uniform_sampling(surface, sample_size)

    # 计算所有采样点对之间的欧氏距离
    for point_a in p:
        for point_b in p:
            distance = np.sqrt(np.sum(np.square(point_b - point_a)))
            distance_list.append(distance)

    # 计算模型对角线的最大距离
    max_distance = math.sqrt(np.sum(np.array(model.shape) ** 2))

    return distance_list, max_distance


def uniform_sampling(array, sample_count: int):
    """
    对输入数组进行均匀采样。

    Args:
        array (np.ndarray): 输入的数组，从中进行采样。
        sample_count (int): 要采样的数量。

    Returns:
        np.ndarray: 采样后的数组。
    """
    length = len(array)
    # 如果数组长度小于等于采样数量，直接返回原数组
    if length <= sample_count:
        return array
    
    # 计算采样间隔
    interval = length / sample_count

    # 初始化采样列表
    sample_list = list()
    for i in range(sample_count):
        # 计算采样索引
        index = (int)(interval * i)
        # 将对应索引的元素添加到采样列表中
        sample_list.append(array[index])

    # 将采样列表转换为 NumPy 数组并返回
    return np.array(sample_list)

def list_to_distribution(array, bins, max_value):
    """
    将输入数组转换为概率分布。

    Args:
        array (list or np.ndarray): 输入的数值数组。
        bins (int): 直方图的区间数量。
        max_value (int or float): 直方图的最大值。

    Returns:
        np.ndarray: 概率分布数组。
    """
    hists, _ = np.histogram(array, bins, range=(0, max_value))

    if np.sum(hists) == 0:
        distribution = np.zeros_like(hists)
    else:
        distribution = hists / np.sum(hists)
    
    return distribution





