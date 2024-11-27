import numpy as np

from voxel_array import open_operator 

def connected_area(model: np.array, type=6):
    model = open_operator(model)

    max_tag = 1
    equals = []

    padding_model = np.zeros(np.array(model.shape) + 2)
    padding_model[1:-1, 1:-1, 1:-1] = model 

    voxels = np.argwhere(padding_model == 1)

    area = np.zeros_like(padding_model)
    for x, y, z in voxels:
        if type == 6:
            neighbor = np.array([area[p] for p in six_neighbor((x, y, z))])
        elif type == 26:
            neighbor = area[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2]
    
        if np.max(neighbor) == 0:
            area[x, y, z] = max_tag

            equals.append(set([max_tag]))
            max_tag = max_tag + 1
        else:
            tag = np.min(neighbor[neighbor > 0])
            area[x, y, z] = tag

            for equal in equals:
                if tag in equal:
                    equal.update(neighbor[neighbor > tag])
                    break
    
    for x, y, z in voxels:
        tag = area[x, y, z]
        for equal in equals:
            if tag in equal:
                area[x, y, z] = min(min(equal), area[x, y, z])
    
    temp = area[1:-1, 1:-1, 1:-1]

    # result = np.zeros_like(temp)

    # value_set = list(set(temp.flatten()))
    # for index in range(len(value_set)):
    #     result[temp == value_set[index]] = index
    return temp

        

            


def six_neighbor(p: tuple[int, int, int]) -> list:
    x, y, z = p
    return[
        (x - 1, y, z),
        (x + 1, y, z),
        (x, y - 1, z),
        (x, y + 1, z),
        (x, y, z - 1),
        (x, y, z + 1),
    ]
    