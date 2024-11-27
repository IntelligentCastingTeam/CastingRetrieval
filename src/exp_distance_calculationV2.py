import os
from tqdm import tqdm
import numpy as np
import json
import traceback

from MathDescription import cal_hull_centroid_percent, cal_cavity_centroid_percent, cal_volume
from ShapeDescription import sample_n2_distance

hull_path = './data/fill_inside'
cavities_path = './data/labeled_cavities'

save_partCode_path = './data/partCode_distance'
if not os.path.exists(save_partCode_path):
    os.mkdir(save_partCode_path)

for file in tqdm(os.listdir(hull_path)):
    name = os.path.splitext(file)[0]

    try:
        hull = np.load(os.path.join(hull_path, file))
        cavities = np.load(os.path.join(cavities_path, file))

        code = {}
        hull_centroid, p_min, p_max = cal_hull_centroid_percent(hull)
        print(hull_centroid)
        code["centroid"] = hull_centroid.tolist()
        code["volume"] = cal_volume(hull)
        distance_list, max_distance = sample_n2_distance(hull, name)
        code["max_distance"] = max_distance
        code["distances"] = distance_list       

        cavity_values = set(cavities.flatten().tolist())
        print(f"零件{name}的cavity_values是", cavity_values)

        cavity_codes = []
        for value in cavity_values:
            if value == 0:
                continue      
            cavity = np.zeros_like(cavities)
            cavity[cavities == value] = 1
            cavity_code = {}
            cavity_code["centroid"] = cal_cavity_centroid_percent(cavity, p_min, p_max).tolist()
            cavity_code["volume"] = cal_volume(cavity)
            cavity_distance_list, cavity_max_distance = sample_n2_distance(cavity, name)
            cavity_code["max_distance"] = cavity_max_distance
            cavity_code['distances'] = cavity_distance_list
            cavity_codes.append(cavity_code)
        
        code["cavity_codes"] = cavity_codes

            

        with open(os.path.join(save_partCode_path, f"{name}.json"), 'w') as json_file:
            json.dump(code, json_file)
         

    except Exception:
        print(f"{name} extraction distance error: {traceback.format_exc()}")


