import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import json

partCode_distance_path = './data/partCode_distance'
partCode_n2_path = './data/partCode_n2'
if not os.path.exists(partCode_n2_path):
    os.mkdir(partCode_n2_path)


for file in tqdm(os.listdir(partCode_distance_path)):
    with open(os.path.join(partCode_distance_path, file), 'r') as json_file:
        code = json.load(json_file)

    distances = code['distances']
    max_distance = code['max_distance']
    
    bin_counts, bin_edges = np.histogram(distances, bins=120, range=(0, max_distance))
    bin_counts = bin_counts / np.sum(bin_counts)
    code['n2'] = bin_counts.tolist()
    # 删除 "distances" 字段
    if "distances" in code:
        del code["distances"]

    for cavity_code in code["cavity_codes"]:
        cavity_max_distance = cavity_code['max_distance']
        cavity_distances = cavity_code['distances']

        bin_counts, bin_edges = np.histogram(cavity_distances, bins=120, range=(0, cavity_max_distance)) 
        bin_counts = bin_counts / np.sum(bin_counts)
        cavity_code['n2'] = bin_counts.tolist()

        if "distances" in cavity_code:
            del cavity_code["distances"]

    with open(os.path.join(partCode_n2_path, file), 'w') as new_json_file:
        json.dump(code, new_json_file)
 










