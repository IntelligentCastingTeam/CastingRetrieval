import os
import numpy as np

import json
import math
from scipy.stats import entropy

distance_counts_path = './data/distance_counts.json'

name1 = 'tutai01'
name2 = 'nippon_thompson_LWH20R_2'

with open(distance_counts_path, 'r') as fp:
    code = json.load(fp)

distance_counts_01 = np.array(code[name1])
distance_counts_02 = np.array(code[name2])

distance_counts_01 = distance_counts_01 / np.sum(distance_counts_01)
distance_counts_02 = distance_counts_02 / np.sum(distance_counts_02)

epsilon = 1e-10
distance_counts_01 = np.where(distance_counts_01 == 0, epsilon, distance_counts_01)
distance_counts_02 = np.where(distance_counts_02 == 0, epsilon, distance_counts_02)

print(np.sum(distance_counts_01))
print(np.sum(distance_counts_02))

kl_divergence = entropy(distance_counts_01, distance_counts_02)
sim = 1 / (1 + kl_divergence)

print(f"凸包{name1}与凸包{name2}之间的KL散度为{kl_divergence}")
print(f"凸包{name1}与凸包{name2}之间的相似度为{sim}")


