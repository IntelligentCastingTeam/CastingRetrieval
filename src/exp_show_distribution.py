import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import json

distances_path = './data/distances'
distance_distribution_path = './data/distance_distribution'
max_distance_path = './data/max_distances.json'
distance_counts_path = './data/distance_counts.json'

# 设置plt字体格式
rcParams['font.sans-serif'] = ['SimHei']
# rcParams['xtick.labelsize'] = 20
# rcParams['ytick.labelsize'] = 20

distance_counts = {}

for file in tqdm(os.listdir(distances_path)):
    name = os.path.splitext(file)[0]

    distances = []
    with open(os.path.join(distances_path, file), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                distances.append(float(row[0].strip()))
            except ValueError:
                pass    # 跳过无法转换为浮点数的行

    if not distances:
        raise ValueError("No valid distance data found in the input file.")

    distances = np.array(distances)
    with open(max_distance_path, 'r') as fp:
        code = json.load(fp)
    
    max_distance = code[name]
    # print(max_distance)

    bin_counts, bin_edges = np.histogram(distances, bins=120, range=(0, max_distance))
    bin_counts = bin_counts / np.sum(bin_counts)
    bin_edges = bin_edges / max_distance

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, bin_counts, width=bin_width)
    # plt.title(f'{name}的D2距离分布直方图')
    # plt.xlabel('归一化点对距离 d')
    # plt.ylabel('距离分布频率 f')
    # plt.show()
    plt.xticks(fontsize=36, fontname='Times New Roman')
    plt.yticks([0.005, 0.010, 0.015, 0.020, 0.025], fontsize=36, fontname='Times New Roman')

    plt.savefig(os.path.join(distance_distribution_path, f"{name}_histogram.png"))
    plt.close()

    distance_counts[f"{name}"] = bin_counts.tolist()

with open(distance_counts_path, 'w') as json_file:
    json.dump(distance_counts, json_file)










