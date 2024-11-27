import os
import numpy as np
from tqdm import tqdm
import traceback

from fill_inside import fill_inside

voxelModel_path = './data/voxelModel'
hull_path = './data/fill_inside'

for file in tqdm(os.listdir(voxelModel_path)):
    name = os.path.splitext(file)[0]

    file_path = os.path.join(voxelModel_path, file)

    try:
        print(f"{name} fill inside start")

        model = np.load(file_path)

        hull = fill_inside(model)

        np.save(os.path.join(hull_path, name), hull)

        print(f"{name} fill inside finished")

    except Exception:
        print(f'{name} fill inside error: {traceback.format_exc()}')



