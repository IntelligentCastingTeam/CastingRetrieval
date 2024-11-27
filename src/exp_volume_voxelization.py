import os
import numpy as np
import logging
from tqdm import tqdm
import traceback
from cell_type import Model
from voxel_array import voxelization

stlModel_path = './data/stlModel'
voxelModel_path = './data/voxelModel'

size = np.array([64, 64, 64])

if not os.path.exists(voxelModel_path):
    os.mkdir(voxelModel_path)

for file in tqdm(os.listdir(stlModel_path)):
    name = os.path.splitext(file)[0]

    file_path = os.path.join(stlModel_path, file)

    try:
        print(f"{name} volume voxelization started")

        model = Model(file_path)
        geometric = np.array([model.length, model.width, model.height])

        #offset = np.ceil(np.max(geometric / size) * 100) / 100
        offset = np.ceil(np.max(geometric) / 64 * 100) / 100
        print("各方向体素数量：", geometric/offset)
        #print(offset)

        voxel = voxelization(model, offset)

        np.save(os.path.join(voxelModel_path, name), voxel)

    except Exception:
        print(f'{name} volume voxelization error: {traceback.format_exc()}')
    
print('volume voxelization finished')



