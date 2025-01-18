import os
import logging
import traceback
from tqdm import tqdm
import numpy as np
import csv
import json

from ShapeDescription import sample_n2_distance

hull_path = './data/fill_inside'
cavities_path = './data/labeled_cavities'

distances_path = './data/distances'
max_distance_path = './data/max_distances.json'



max_distances = {}

for file in tqdm(os.listdir(hull_path)):
    name = os.path.splitext(file)[0]

    try:
        hull = np.load(os.path.join(hull_path, file))

        logging.info(f"{name} start")

        distance_list, max_distance = sample_n2_distance(hull, name)

        output_filename = f"{name}.csv"
        with open(os.path.join(distances_path, output_filename), 'w', newline = "") as file:
            writer = csv.writer(file)

            for item in distance_list:
                writer.writerow([item])
        
        max_distances[f'{name}'] = max_distance

    except Exception:
        logging.error(f"{name} extraction distance error: {traceback.format_exc()}")

with open(max_distance_path, 'w') as json_file:
    json.dump(max_distances, json_file)
