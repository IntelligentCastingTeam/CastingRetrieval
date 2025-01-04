import numpy as np
import random
import math


def sample_voxel_N2(
    model: np.ndarray, output_dimension, shape=None, sample_size=512, neighbor=6
):
    """ """
    if neighbor == 6:
        surface_voxel = get_6_neigbor_surface(model)
    else:
        surface_voxel = get_26_neighbor_suface(model)

    distance_list = []
    surface = np.argwhere(surface_voxel == 2)
    if len(surface) <= sample_size:
        p1 = surface
        p2 = surface
    else:
        p1 = random.sample(list(surface), min(len(surface), sample_size))
        p2 = random.sample(list(surface), min(len(surface), sample_size))
    for point_a in p1:
        for point_b in p2:
            distance = np.sqrt(np.sum(np.square(point_b - point_a)))
            distance_list.append(distance)

    if shape is not None:
        max_distance = math.sqrt(np.sum(np.array(shape) ** 2))
    else:
        max_distance = math.sqrt(np.sum(np.array(model.shape) ** 2))

    distribution = list_to_distribution(distance_list, output_dimension, max_distance)

    return distribution


def list_to_distribution(array, bins, max_value):
    hists, _ = np.histogram(array, bins=range(bins), range=(0, max_value))

    if np.sum(hists) == 0:
        distribution = np.zeros_like(hists)
    else:
        distribution = hists / np.sum(hists)

    return distribution


def get_6_neigbor_surface(model: np.ndarray):
    if len(model.shape) != 3:
        ValueError("input model is not three-dimension array")

    padding_model = np.zeros(np.array(model.shape) + 2)
    padding_model[1:-1, 1:-1, 1:-1] = model
    for x, y, z in np.argwhere(padding_model >= 1):
        if np.min(get_neigbor_six(padding_model, x, y, z)) == 0:
            padding_model[x, y, z] = 2
        else:
            padding_model[x, y, z] = 1

    return padding_model[1:-1, 1:-1, 1:-1]


def get_neigbor_six(model: np.ndarray, x, y, z):
    neigbor = [
        model[x - 1, y, z],
        model[x + 1, y, z],
        model[x, y - 1, z],
        model[x, y + 1, z],
        model[x, y, z - 1],
        model[x, y, z + 1],
    ]

    return np.array(neigbor)


def get_26_neighbor_suface(model: np.ndarray):
    """
    get suface voxel from a voxel set.

    parameters:
    ========================================
    model: a three dimension array of voxels
    ========================================

    output:
    ========================================
    numpy.ndarray: a three dimension array of surface voxels
                   0 presents void
                   1 presents inside voxel
                   2 presents surface voxel
    ========================================
    """

    if len(model.shape) != 3:
        ValueError("input model is not three-dimension array")

    padding_model = np.zeros(np.array(model.shape) + 2)
    padding_model[1:-1, 1:-1, 1:-1] = model

    for x, y, z in np.argwhere(padding_model > 0):
        if np.min(padding_model[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2]) == 0:
            padding_model[x, y, z] = 2
        else:
            padding_model[x, y, z] = 1

    return padding_model[1:-1, 1:-1, 1:-1]
