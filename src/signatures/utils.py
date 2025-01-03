import numpy as np
from scipy.stats import entropy


def cal_simulation(first_code: dict, second_code: dict):
    volume_distance = abs(first_code["volume"] - second_code["volume"])
    hull_sim = distance_to_similarity(
        KLdivergence(np.array(first_code["code"]), np.array(second_code["code"]))
    )

    cav_sim = 0
    if len(first_code["cavity_codes"]) == 0 or len(second_code["cavity_codes"]) == 0:
        cav_sim = 0
    else:
        sim_array = np.zeros(
            [len(first_code["cavity_codes"]), len(second_code["cavity_codes"])]
        )

        for x, first_cavity in zip(
            range(len(first_code["cavity_codes"])), first_code["cavity_codes"]
        ):
            for y, second_cavity in zip(
                range(len(second_code["cavity_codes"])), second_code["cavity_codes"]
            ):
                sim = distance_to_similarity(
                    KLdivergence(
                        np.array(first_cavity["code"]), np.array(second_cavity["code"])
                    )
                )
                pos_sim = distance_to_similarity(
                    euler_distance(
                        np.array(first_cavity["centroid"]),
                        np.array(second_cavity["centroid"]),
                    )
                )
                vol_sim = 1 - abs(first_cavity["volume"] - second_cavity["volume"]) / (
                    first_cavity["volume"] + first_cavity["volume"]
                )
                sim_array[x, y] = sim * pos_sim

        cav_sim = np.average(np.max(sim_array, axis=1))
    if cav_sim == 0:
        sim = hull_sim
    else:
        sim = 2 * hull_sim * cav_sim / (hull_sim + cav_sim)

    return {
        "hull_sim": hull_sim,
        "cavity_sim": cav_sim,
        "sim": sim,
    }
    pass


def distance_to_similarity(distance: np.ndarray) -> np.ndarray:
    return 1 / (1 + distance)


def KLdivergence(a: np.ndarray, b: np.ndarray):
    kl_div = entropy(a, b)
    return kl_div


def euler_distance(a: np.ndarray, b: np.ndarray):
    distance = np.linalg.norm(a - b)
    return distance
