import numpy as np
from cell_type import Model
import cv2

def get_6_neighbor_surface(model: np.ndarray):
    """
    获取三维模型的表面体素，使用 6 邻域判定。

    Args:
        model (np.ndarray): 输入的三维体素模型。

    Returns:
        np.ndarray: 标记表面体素的三维模型，其中表面体素标记为 2，内部体素标记为 1。
    """
    if len(model.shape) != 3:
        ValueError("input model is not three-dimension array")

    padding_model = np.zeros(np.array(model.shape) + 2)
    padding_model[1:-1, 1:-1, 1:-1] = model

    indicesList = np.argwhere(padding_model >= 1)
    for x, y, z in indicesList:
        # 如果当前体素的6邻域内包含0，则标记为表面体素（2）
        if np.min(get_neighbor_six(padding_model, x, y, z)) == 0:
            padding_model[x, y, z] = 2
        else:
            # 否则标记为内部体素（1）
            padding_model[x, y, z] = 1
    # 去除填充的边界，返回结果
    return padding_model[1:-1, 1:-1, 1:-1]



def get_neighbor_six(model: np.ndarray, x, y, z):
    neighbor = [
        model[x - 1, y, z],
        model[x + 1, y, z],
        model[x, y - 1, z],
        model[x, y + 1, z],
        model[x, y, z - 1],
        model[x, y, z + 1],
    ]

    return np.array(neighbor)

def voxelization(model: Model, offset):
    shape = [(int)(model.width / offset + 1), (int)(model.length / offset + 1)]
    slices = []

    for z in np.arange(model.min_z + 0.5 * offset, model.max_z, offset):
        img = np.zeros(shape, dtype=np.uint8)
        lines = model.get_slice(z)

        for line in lines:
            x_0 = (int)((line.start.x - model.min_x) / offset)
            y_0 = (int)((line.start.y - model.min_y) / offset)

            x_1 = (int)((line.end.x - model.min_x) / offset)
            y_1 = (int)((line.end.y - model.min_y) / offset)

            cv2.line(img, (x_0, y_0), (x_1, y_1), 1, 1)
        
        for y in np.arange(model.min_y + 0.5 * offset, model.max_y, offset):
            trance_points = []
            for line in lines:
                if line.start.y < y and y < line.end.y:
                    point_y = y
                    point_x = (line.end.x - line.start.x) * ((y - line.start.y) / (line.end.y - line.start.y)) + line.start.x
                    trance_points.append([point_x, point_y])

            if len(trance_points) > 0:
                sorted_points = sorted(trance_points, key=lambda x: x[0])
                for index in range(len(sorted_points)):
                    if index % 2 == 0:
                        p_0 = sorted_points[index]
                        if index + 1 < len(sorted_points):
                            p_1 = sorted_points[index + 1]
                        else:
                            p_0 = p_1
                        
                        x_0 = (int)((p_0[0] - model.min_x) / offset)
                        y_0 = (int)((p_0[1] - model.min_y) / offset)

                        x_1 = (int)((p_1[0] - model.min_x) / offset)
                        y_1 = (int)((p_1[1] - model.min_y) / offset)

                        cv2.line(img, (x_0, y_0), (x_1, y_1), 1, 1)                        

        slices.append(img)
    
    voxel_model = np.array(slices)
    voxel_model = voxel_model.transpose([2, 1, 0])

    return voxel_model


def open_operator(voxel: np.ndarray) -> np.ndarray:
    """
    对三维体素模型进行开运算（先腐蚀后膨胀），以去除噪声和小物体。

    Args:
        voxel (np.ndarray): 输入的三维体素模型。

    Returns:
        np.ndarray: 经过开运算处理后的三维体素模型。
    """
    # 增加模型的维度，防止边界问题
    new_voxel_shape = np.array(voxel.shape) + 4
    new_voxel = np.zeros(new_voxel_shape)
    new_voxel[2:-2, 2:-2, 2:-2] = voxel

    # 初始化目标数组
    target_0 = np.zeros(new_voxel_shape)
    # 腐蚀操作：遍历所有非零体素
    for x, y, z in np.argwhere(new_voxel > 0):
        # 如果当前体素的3x3x3邻域内包含0，则设置当前体素为0
        if np.min(new_voxel[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2]) == 0:
            target_0[x, y, z] = 0
        else:
            target_0[x, y, z] = 1

    # 初始化目标数组
    target = np.zeros(new_voxel_shape)
    # 膨胀操作：遍历所有非零体素
    for x, y, z in np.argwhere(target_0 > 0):
        # 将当前体素的3x3x3邻域内的体素设置为1
        target[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2] = target_0[x, y, z]

    # 去除填充的边界，恢复原始模型的形状
    result = target[2:-2, 2:-2, 2:-2]
    return result