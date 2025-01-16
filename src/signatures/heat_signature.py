import numpy as np
from scipy.ndimage import convolve
from sklearn.cluster import KMeans


def heat_signature(
    model: np.ndarray, dim: int, steps: int = 32, alpha: float = 0.1
) -> np.ndarray:
    """Compute the heat signature fro all points

    Args:
        dim (int): Dimensionality (timesteps) of the signature
    """

    kernel = np.zeros([3, 3, 3])
    kernel[1, 1, 1] = -6
    kernel[0, 1, 1] = 1
    kernel[1, 0, 1] = 1
    kernel[1, 1, 0] = 1
    kernel[2, 1, 1] = 1
    kernel[1, 2, 1] = 1
    kernel[1, 1, 2] = 1

    tempature = np.copy(model).astype(np.float32)
    ps = np.argwhere(tempature > 0)

    if len(ps) < dim * 4:
        min_x, min_y, min_z = np.min(ps, axis=1)
        max_x, max_y, max_z = np.max(ps, axis=1)
        tempature = tempature[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1]
        tempature = np.repeat(tempature, 4, axis=0)
        tempature = np.repeat(tempature, 4, axis=1)
        tempature = np.repeat(tempature, 4, axis=2)
        alpha = alpha / 16

    mask = np.zeros_like(tempature, dtype=np.int8)
    mask[tempature > 0] = 1
    tempature[tempature > 0] = 1
    tempature = tempature * 1000
    points = np.argwhere(tempature == 1000)
    values = []

    for _ in range(steps):
        tempature = tempature + alpha * convolve(
            tempature, weights=kernel, mode="constant"
        )
        value = [tempature[x, y, z] for x, y, z in points]
        values.append(value)
        tempature = tempature * mask
    values = np.array(values).T

    kmeans = KMeans(
        n_clusters=dim, random_state=0, init="k-means++", n_init="auto"
    ).fit(values)
    result = kmeans.predict(values)

    centers = np.mean(kmeans.cluster_centers_, axis=1)
    sorted_labels = sorted(range(dim), key=lambda k: centers[k])
    sorted_result = []
    for i in result:
        sorted_result.append(sorted_labels.index(i))

    code = np.bincount(sorted_result, weights=None, minlength=dim)
    code = code.astype(np.float32) / np.sum(code)

    return code
    pass
