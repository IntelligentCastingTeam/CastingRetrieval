import os 
import numpy as np
from tqdm import tqdm
import traceback
from extract_code import connected_area

hull_path = './data/fill_inside'
labeled_cavities_path = './data/labeled_cavities'

for file in tqdm(os.listdir(hull_path)):
    name = os.path.splitext(file)[0]

    file_path = os.path.join(hull_path, file)

    try:
        print(f"{name} cavities labeling started")

        model = np.load(file_path)
        cavities = np.zeros_like(model)
        cavities[model == 3] = 1

        labeled_cavities = connected_area(cavities)

        np.save(os.path.join(labeled_cavities_path, name), labeled_cavities)



        print(f"{name} cavities labeling finished")
    except Exception:
        print(f'{name} cavities labeling error: {traceback.format_exc()}')


