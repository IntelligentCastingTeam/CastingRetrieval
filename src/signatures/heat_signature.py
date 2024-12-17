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

    tempature = model.astype(np.float32)
    mask = np.zeros_like(model)
    mask[model > 0] = 1
    tempature[tempature > 0] = 1
    tempature = tempature * 1000
    points = np.argwhere(tempature == 1000)
    values = []

    for _ in range(steps):
        tempature = tempature + alpha * convolve(tempature, weights=kernel)
        value = [tempature[x, y, z] for x, y, z in points]
        values.append(value)
        tempature = tempature * mask
    values = np.array(values).T

    kmeans = KMeans(
        n_clusters=dim, random_state=0, init="k-means++", n_init="auto"
    ).fit(values)
    result = kmeans.predict(values)

    code = np.bincount(result, weights=None, minlength=dim)
    code = code.astype(np.float32) / np.sum(code)

    return code
    pass
