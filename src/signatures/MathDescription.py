import math

import numpy as np
from deprecated import deprecated


def cal_volume(model: np.ndarray) -> int:
    points = np.argwhere(model > 0)
    return len(points)


def cal_centroid_moment_mat(model: np.ndarray) -> [np.ndarray, float, float]:
    """
    计算三维的二阶矩
    :param model:
    :return
    """
    binary_model = np.zeros(model.shape)
    binary_model[model > 0] = 1

    M000 = np.sum(binary_model)
    list = np.argwhere(binary_model == 1)
    x_avenger = sum(point[0] for point in list) / M000
    y_avenger = sum(point[1] for point in list) / M000
    z_avenger = sum(point[2] for point in list) / M000

    mu_200 = sum((point[0] - x_avenger) ** 2 for point in list) / len(list)
    mu_020 = sum((point[1] - y_avenger) ** 2 for point in list) / len(list)
    mu_002 = sum((point[2] - z_avenger) ** 2 for point in list) / len(list)
    mu_110 = sum(
        (point[0] - x_avenger) * (point[1] - y_avenger) for point in list
    ) / len(list)
    mu_011 = sum(
        (point[1] - y_avenger) * (point[2] - z_avenger) for point in list
    ) / len(list)
    mu_101 = sum(
        (point[0] - x_avenger) * (point[2] - z_avenger) for point in list
    ) / len(list)

    mu_mat = np.array(
        [[mu_200, mu_110, mu_101], [mu_110, mu_020, mu_011], [mu_101, mu_011, mu_002]]
    )

    j1 = mu_200 + mu_020 + mu_002
    j2 = (
        mu_200 * mu_020
        - mu_110**2
        + mu_020 * mu_002
        - mu_011**2
        + mu_200 * mu_002
        - mu_101**2
    )
    delta2 = np.linalg.det(mu_mat)
    return [mu_mat, j2 / (j1**2), delta2 / (j1**3)]


def cal_center_moment_mat(model: np.ndarray) -> [np.ndarray, float, float]:
    """
    计算三维的二阶矩
    :param model:
    :return
    """
    binary_model = np.zeros(model.shape)
    binary_model[model > 0] = 1

    M000 = np.sum(binary_model)
    list = np.argwhere(binary_model == 1)
    x_avenger = model.shape[0] / 2
    y_avenger = model.shape[1] / 2
    z_avenger = model.shape[2] / 2

    mu_200 = sum((point[0] - x_avenger) ** 2 for point in list)
    mu_020 = sum((point[1] - y_avenger) ** 2 for point in list)
    mu_002 = sum((point[2] - z_avenger) ** 2 for point in list)
    mu_110 = sum((point[0] - x_avenger) * (point[1] - y_avenger) for point in list)
    mu_011 = sum((point[1] - y_avenger) * (point[2] - z_avenger) for point in list)
    mu_101 = sum((point[0] - x_avenger) * (point[2] - z_avenger) for point in list)

    mu_mat = np.array(
        [[mu_200, mu_110, mu_101], [mu_110, mu_020, mu_011], [mu_101, mu_011, mu_002]]
    )

    j1 = mu_200 + mu_020 + mu_002
    j2 = (
        mu_200 * mu_020
        - mu_110**2
        + mu_020 * mu_002
        - mu_011**2
        + mu_200 * mu_002
        - mu_101**2
    )
    delta2 = np.linalg.det(mu_mat)
    return [mu_mat, j2 / (j1**2), delta2 / (j1**3)]


@deprecated
def count_connected_area(model: np.ndarray):
    max_tag = 1
    equals = []
    new_shape = np.array(model.shape) + 2
    padding_model = np.zeros(new_shape)
    padding_model[1:-1, 1:-1, 1:-1] = model
    area = np.zeros(padding_model.shape)
    voxel_model = np.argwhere(padding_model == 1)
    for x, y, z in voxel_model:
        neighbor = area[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2]
        if np.max(neighbor) == 0:
            area[x, y, z] = max_tag
            equals.append(set([max_tag]))
            max_tag = max_tag + 1
        else:
            tag = np.min(neighbor[neighbor > 0])
            area[x, y, z] = tag
            for equal in equals:
                if tag in equal:
                    equal.update(neighbor[neighbor > tag])
                    break

    for x, y, z in voxel_model:
        tag = area[x, y, z]
        for equal in equals:
            if tag in equal:
                area[x, y, z] = min(min(equal), area[x, y, z])

    result = area[1:-1, 1:-1, 1:-1]
    return result, len(set(area[area > 0]))


def cal_center_percent(model: np.ndarray) -> np.ndarray:
    return np.array([0.5, 0.5, 0.5])


def cal_centroid_pos(model: np.ndarray) -> np.ndarray:
    point_list = np.argwhere(model > 0)
    centroid = np.sum(point_list, axis=0) / len(point_list)

    return centroid


def cal_centroid_percent(model: np.ndarray) -> np.ndarray:
    """
    计算模型质心相对于自身的坐标
    :param model:
    :return: [x%, y%, z%]
    """
    point_list = np.argwhere(model > 0)
    centroid = np.sum(point_list, axis=0) / len(point_list)

    p_max = np.max(point_list, axis=0)
    p_min = np.min(point_list, axis=0)
    precent = (centroid - p_min + 1) / (p_max - p_min + 1)

    return precent


def cal_centroid(model: np.ndarray):
    # model[model > 0] = 1
    m = np.sum(model)
    point_list = np.argwhere(model >= 1)
    centroid = np.sum(point_list, axis=0) / m

    return centroid


if __name__ == "__main__":
    model = np.zeros([50, 50, 50])
    model[3:10, 3:10, 3:10] = 1
    model[11:20, 11:20, 11:20] = 1
    model[22:25, 22:25, 22:25] = 1
    voxel = np.argwhere(model == 1)
    print(voxel)
    s = set(model[model > 0])
    s.update([1, 2, 3, 4])
    _, count = count_connected_area(model)
    print(count)

    pass
