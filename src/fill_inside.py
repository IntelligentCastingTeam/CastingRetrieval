import os
import numpy as np
import cv2

def fill_inside(model: np.ndarray) -> (np.ndarray):
    if len(model.shape) != 3:
        ValueError("input model is not a 3-dimension array")
    result_array = model.copy()

    # 沿 y 和 z 方向处理每一个 x 切片
    for y in range(result_array.shape[1]):
        for z in range(result_array.shape[2]):
            line = result_array[:, y, z]
            fix = np.argwhere(line > 0)
            if len(fix) == 0:
                continue
            _min, _max = np.min(fix), np.max(fix)

            for index in range(_min, _max):
                if result_array[index, y, z] == 0:
                    result_array[index, y, z] == 3

    # 沿 x 和 z 方向处理每一个 y 切片
    for x in range(result_array.shape[0]):
        for z in range(result_array.shape[2]):
            line = result_array[x, :, z]
            fix = np.argwhere(line > 0)
            if len(fix) == 0:
                continue
            _min, _max = np.min(fix), np.max(fix)
            for index in range(_min, _max):
                if result_array[x, index, z] == 0:
                    result_array[x, index, z] = 3

    # 沿 x 和 y 方向处理每一个 z 切片
    for x in range(result_array.shape[0]):
        for y in range(result_array.shape[1]):
            line = result_array[x, y, :]
            fix = np.argwhere(line > 0)
            if len(fix) == 0:
                continue
            _min, _max = np.min(fix), np.max(fix)
            for index in range(_min + 1, _max):
                if result_array[x, y, index] == 0:
                    result_array[x, y, index] = 3
    
    return result_array
    

