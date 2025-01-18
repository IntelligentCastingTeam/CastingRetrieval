import os
import numpy as np
import json
from scipy.stats import entropy

from feature_code import cal_similarity

partCode_n2_path = './data/partCode_n2'

name1 = 'EX33'
name2 = 'HCH_PRT'

part1 = {}
part2 = {} 

with open(os.path.join(partCode_n2_path, f"{name1}.json"), 'r') as fp1:
    part1 = json.load(fp1)
with open(os.path.join(partCode_n2_path, f"{name2}.json"), 'r') as fp2:
    part2 = json.load(fp2)

hull_shape_sim, cavity_sim, total_sim = cal_similarity(part1, part2)

print(f"零件{name1}与零件{name2}之间的外形相似度为{hull_shape_sim}")
print(f"零件{name1}与零件{name2}之间的内腔相似度为{cavity_sim}")
print(f"零件{name1}与零件{name2}之间的总体相似度为{total_sim}")


