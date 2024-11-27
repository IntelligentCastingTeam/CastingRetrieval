import numpy as np
import open3d as o3d

class O3dModel:
    def __init__(self, file_path: str):
        # 使用Open3D库读取三角网格模型
        self._model: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(file_path)
        # 将模型平移,使其中心位于原点
        self._model.translate(-self._model.get_center())
        self._d_mean = 0

        self.__update()


    def __update(self):
        """
        更新模型的顶点、三角面片索引、法向量等属性
        """
        # 获得模型的顶点坐标数组
        self._points = np.asarray(self._model.vertices)
        # 获得三角面片索引数组
        self._triangles = np.asarray(self._model.triangles)
        # 获得三角面片法向量
        self._model.compute_triangle_normals()
        self._normals = np.asarray(self._model.triangle_normals)
        # 获得每个三角面片的三个顶点坐标
        triangle_points = self._points[self._triangles]
        # 计算每个三角面片的质心坐标
        triangle_centroid = np.average(triangle_points, axis=1)
        # 计算每个三角面片质心到原点的平均距离
        self._d_mean = np.average(np.linalg.norm(triangle_centroid, ord=2, axis=1))

    @property
    def points(self) -> np.ndarray:
        return self._points
    
    @property
    def triangles(self) -> np.ndarray:
        return self._triangles
    
    @property
    def normals(self) -> np.ndarray:
        return self._normals
    
    @property
    def max_x(self) -> float:
        return self._model.get_max_bound()[0]

    @property
    def max_y(self) -> float:
        return self._model.get_max_bound()[1]

    @property
    def max_z(self) -> float:
        return self._model.get_max_bound()[2]

    @property
    def min_x(self) -> float:
        return self._model.get_min_bound()[0]

    @property
    def min_y(self) -> float:
        return self._model.get_min_bound()[1]

    @property
    def min_z(self) -> float:
        return self._model.get_min_bound()[2]
    
    @property
    def area(self) -> float:
        return self._model.get_surface_area()
    
    def rotate(self, rotate_mat: np.ndarray, center=np.zeros([3])):
        self._model.rotate(rotate_mat, center)
        self.__update()

    def transform(self, transform_mat: np.ndarray):
        self._model.transform(transform_mat)
        self.__update()
        pass
