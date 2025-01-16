import numpy as np
from scipy.stats import entropy

def cal_similarity(first_code: dict, second_code: dict):
    # # 计算 hull 的 volume 相似度
    # volume_sim = 1 - (abs(first_code['volume'] - second_code['volume']) / 
    #                      (first_code['volume'] + second_code['volume']))

    # 计算 hull 的 n2 相似度
    distance_counts_01 = np.array(first_code["n2"])
    distance_counts_02 = np.array(second_code["n2"])
    hull_shape_sim = cal_shape_sim(distance_counts_01, distance_counts_02)

    cavity_sim = 0
    if len(first_code["cavity_codes"]) != 0 and len(second_code["cavity_codes"]) == 0:
        cavity_sim = 0
    elif len(first_code["cavity_codes"]) == 0 and len(second_code["cavity_codes"]) != 0:
        cavity_sim = 0
    elif len(first_code["cavity_codes"]) == 0 and len(second_code["cavity_codes"]) == 0:
        cavity_sim = 1
    else:
        first_cavity_sim_score = 0
        first_cavity_total_volume = 0
        for first_cavity in first_code["cavity_codes"]:
            shape_total_score = 0
            volume_total_score = 0
            for second_cavity in second_code["cavity_codes"]:
                volume1 = first_cavity["volume"]
                volume2 = second_cavity["volume"]
                cavity_volume_sim = 1 - (abs(volume1 - volume2) / (volume1 + volume2))

                centroid1 = np.array(first_cavity["centroid"])
                centroid2 = np.array(second_cavity["centroid"])
                manhattan_distance  = np.sum(np.abs(centroid1 - centroid2))
                cavity_centroid_sim = 1 / (1 + manhattan_distance)

                cavity_shape_sim = cal_shape_sim(
                    np.array(first_cavity["n2"]),
                    np.array(second_cavity["n2"])
                )

                shape_total_score += cavity_centroid_sim * cavity_shape_sim * cavity_volume_sim
                volume_total_score += cavity_volume_sim
            
            first_cavity_sim_score += (shape_total_score / volume_total_score) * first_cavity["volume"]
            first_cavity_total_volume += first_cavity["volume"]
        
        cavity_sim = first_cavity_sim_score / first_cavity_total_volume
    
    beta = 0.8
    # total_sim = 1 / ((1 - beta) * (1/hull_shape_sim) + beta * (1/cavity_sim))
    total_sim = beta * hull_shape_sim + (1 - beta) * cavity_sim

    return hull_shape_sim, cavity_sim, total_sim







    # 计算空腔相似度
    for i, first_cavity in zip(
        range(len(first_code["cavity_codes"])), first_code["cavity_codes"]
    ):
        for j, second_cavity in zip(
            range(len(second_code["cavity_codes"])), second_code["cavity_codes"]
        ):
            volume1 = first_cavity["volume"]
            volume2 = second_cavity["volume"]
            cavity_volume_sim = 1 - (abs(volume1 - volume2) / (volume1 + volume2))

            centroid1 = np.ndarray(first_cavity["centroid"])
            centroid2 = np.ndarray(second_cavity["centroid"])
            manhattan_distance  = np.sum(np.abs(centroid1 - centroid2))
            cavity_centroid_sim = 1 / (1 + manhattan_distance)

            cavity_shape_sim = cal_shape_sim(
                np.ndarray(first_cavity["n2"]),
                np.ndarray(second_cavity["n2"])
            )

            sim_arr
            sim_ij = cavity_centroid_sim * cal_shape_sim


def cal_shape_sim(distance_counts_01: np.ndarray, distance_counts_02: np.ndarray):
    """
    计算 n2 形状相似度
    """
    epsilon = 1e-10
    distance_counts_01 = np.where(distance_counts_01 == 0, epsilon, distance_counts_01)
    distance_counts_02 = np.where(distance_counts_02 == 0, epsilon, distance_counts_02)
    
    kl_divergence = entropy(distance_counts_01, distance_counts_02)
    shape_sim = 1 / (1 + kl_divergence)

    return shape_sim


