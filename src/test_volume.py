import numpy as np
from cell_type import Model
from voxel_array import voxelization

model = Model("F:/workdir/论文/中文文章/stl测试零件2/1012A_.STL")
print(model.volume)
print(model.triangles_count)
print(model.min_x, model.min_y, model.min_z, model.max_x, model.max_y, model.max_z)

geometric = np.array([model.length, model.width, model.height])
offset = np.ceil(np.max(geometric) / 64 * 100) / 100

voxel = voxelization(model, offset)