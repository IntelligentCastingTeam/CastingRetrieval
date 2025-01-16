import numpy as np
from model import O3dModel
import transformations

def CPCA_normalize_Guo(model: O3dModel):
    triangles = model.triangles.tolist()
    points = model.points
    areas = np.array([cal_triangle_area(triangle, points) for triangle in triangles])
    # 计算加权质心
    centroids = (np.array([cal_centroid(triangle, points) for triangle in triangles]).transpose() * areas).transpose()
    # 计算质心矩阵
    moment_mat = np.dot(centroids.T, centroids) / model.area 
    # 特征值分解
    w, v = np.linalg.eig(moment_mat)
    i = np.argsort(-w)
    # 按特征值大小调整特征向量
    v[:, [0, 1, 2]] = v[:, i]
    # 叉乘第一个和第二个特征向量，得到第三个方向
    a = np.cross(v[:, 0], v[:, 1])
    if np.dot(a , v[:, 2]) < 0:
        v[:, 2] = -v[:, 2]
    # 对模型进行旋转，使其主轴与坐标轴对齐
    model.rotate(v.T)
    # 初始化单位矩阵
    T = transformations.identity_matrix()
    # 判断最大和最小值，调整模型姿态使其正向对齐
    if abs(model.max_x) < abs(model.min_x):
        T[0, 0] = -1
    if abs(model.max_y) < abs(model.min_y):
        T[1, 1] = -1
    if abs(model.max_z) < abs(model.min_z):
        T[2, 2] = -1
    model.transform(T)

def CPCA_normalize(model: O3dModel):
    triangles = model.triangles.tolist()
    points = model.points
    areas = np.array([cal_triangle_area(triangle, points) for triangle in triangles])
    # 计算模型总面积
    total_area = np.sum(areas)
    # 计算模型的质心
    centroids = np.array([cal_centroid(triangle, points) for triangle in triangles])
    weighted_centroids = (centroids.T * areas).T
    m_I = np.sum(weighted_centroids, axis=0) / total_area

    # 平移所有顶点
    points_centered = points - m_I

    # 计算协方差矩阵C
    C = np.zeros((3, 3))
    for i in range(len(triangles)):
        # 获取三角形顶点索引
        idx = triangles[i]
        p_A = points_centered[idx[0]]
        p_B = points_centered[idx[1]]
        p_C = points_centered[idx[2]]
        # 计算三角形的面积
        S_i = areas[i]
        # 计算三角形的重心
        g_i = (p_A + p_B + p_C) / 3
        # 计算协方差贡献
        term = (p_A[:, None] * p_A[None, :]) + (p_B[:, None] * p_B[None, :]) + (p_C[:, None] * p_C[None, :]) + 9 * (g_i[:, None] * g_i[None, :])
        C += (S_i / 12) * term
    C = C / total_area

    # 特征值分解
    w, v = np.linalg.eigh(C)
    # 按特征值从大到小排序
    idx = np.argsort(-w)
    v = v[:, idx]
    # 确保右手系
    if np.linalg.det(v) < 0:
        v[:, 2] = -v[:, 2]
    
    model.rotate(v.T)



def cal_triangle_area(triangle, points: np.ndarray):
    ab = points[triangle[1]] - points[triangle[0]]
    ac = points[triangle[2]] - points[triangle[0]]
    return np.linalg.norm(np.cross(ab, ac)) / 2

def cal_centroid(triangle, points: np.ndarray):
    return (points[triangle[0]] + points[triangle[1]] + points[triangle[2]]) / 3
