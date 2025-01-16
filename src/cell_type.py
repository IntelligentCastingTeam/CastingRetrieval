import os
import struct
import numpy as np
from enum import Enum
import transformations

class Axes(Enum):
    X = 0
    Y = 1 
    Z = 2

class Point:
    def __init__(self, buffer=None, x: float = 0, y: float = 0, z: float = 0) -> None:
        if buffer:
            # struct模块将二进制数据解析为浮点数
            self.x, self.y, self.z = struct.unpack('3f', buffer[0:12])
        else:
            self.x = x
            self.y = y
            self.z = z
    
    def transform(self, mat):
        point = np.array([self.x, self.y, self.z, 1])
        point = np.dot(point, mat)
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

    def rotate(self, rotate_mat: np.ndarray, center: np.ndarray):
        point = np.array([self.x, self.y, self.z])
        point = np.dot(rotate_mat, point - center)
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]
        pass


class Line:
    def __init__(self, start: Point, end: Point, normal: np.ndarray) -> None:
        self.start = start
        self.end = end
        self.normal = normal
        self.update_bounds()
    
    def __str__(self):
        return (f"Start: ({self.start.x}, {self.start.y}, {self.start.z}), "
                f"End: ({self.end.x}, {self.end.y}, {self.end.z}),")

    def update_bounds(self):
        self.min_x = min(self.start.x, self.end.x)
        self.max_x = max(self.start.x, self.end.x)
        self.min_y = min(self.start.y, self.end.y)
        self.max_y = max(self.start.y, self.end.y)
        self.sort(Axes.Y)
        pass

    def sort(self, axes: Axes):
        if axes == Axes.X:
            if self.start.x > self.end.x:
                self.start, self.end = self.end, self.start
        elif axes == Axes.Y:
            if self.start.y > self.end.y:
                self.start, self.end = self.end, self.start
        elif axes == Axes.Z:
            if self.start.z > self.end.z:
                self.start, self.end = self.end, self.start
        


class FaceNormal:
    """
    三角面法向量
    """
    def __init__(self, buffer=None, x=0, y=0, z=0) -> None:
        if buffer:
            self.x, self.y, self.z = struct.unpack('3f', buffer[0:12])
        else:
            self.x, self.y, self.z = x, y, z
    
    def transform(self, mat):
        normal = np.array([self.x, self.y, self.z, 0])
        normal = np.dot(normal, mat)
        self.x = normal[0]
        self.y = normal[1]
        self.z = normal[2]       


class Triangle:
    def __init__(self, a: Point=Point(), b: Point=Point(), c:Point=Point(), n:FaceNormal=FaceNormal(), buffer=None) -> None:
        self.min_x = self.min_y = self.min_z = 0
        self.max_x = self.max_y = self.max_z = 0
        if buffer:
            self.__init_by_buffer(buffer)
        else:
            self.a = a
            self.b = b
            self.c = c
            self.normal = n

        self.area = 0.5 * np.linalg.norm(
            np.cross(
                np.array(
                    [self.b.x - self.a.x, self.b.y - self.a.y, self.b.z - self.a.z]
                ),
                np.array(
                    [self.c.x - self.a.x, self.c.y - self.a.y, self.c.z - self.a.z]
                ),
            )
        )

        self.sort()
        self.__update_bounds()

    def __init_by_buffer(self, buffer):
        self.normal = FaceNormal(buffer[0:12])
        self.a = Point(buffer[12:24])
        self.b = Point(buffer[24:36])
        self.c = Point(buffer[36:48])
    
    def sort(self):
        """
        对三角形的顶点按照 Z 轴坐标进行排序 
        a.z < b.z < c.z
        """
        if self.a.z > self.b.z:
            self.a, self.b = self.b, self.a
        if self.a.z > self.c.z:
            self.a, self.c = self.c, self.a
        if self.b.z > self.c.z:
            self.b, self.c = self.c, self.b
    
    def __update_bounds(self):
        self.min_x = min(self.a.x, self.b.x, self.c.x)
        self.max_x = max(self.a.x, self.b.x, self.c.x)
        self.min_y = min(self.a.y, self.b.y, self.c.y)
        self.max_y = max(self.a.y, self.b.y, self.c.y)
        self.min_z = min(self.a.z, self.b.z, self.c.z)
        self.max_z = max(self.a.z, self.b.z, self.c.z)
        self.centroid = [
            (self.a.x + self.b.x + self.c.x) / 3,
            (self.a.y + self.b.y + self.c.y) / 3,
            (self.a.z + self.b.z + self.c.z) / 3,
        ]

    def transform(self, mat):
        self.a.transform(mat)
        self.b.transform(mat)
        self.c.transform(mat)
        self.normal.transform(mat)
        self.sort()
        self.__update_bounds()

    def slice_z(self, z):
        if z < self.min_z or z > self.max_z:
            return None
        else:
            # 如果 z 切割平面通过顶点 A 并且在 B 和 C 之下，则返回 None
            if z == self.a.z and z < self.b.z and z < self.c.z:
                # return [Line(self.a, self.a, np.array([self.normal.x, self.normal.y]))]
                return None
            # 如果 z 切割平面通过顶点 A 和 B 并且在 C 之下，则返回 AB 线段
            elif z == self.a.z and z == self.b.z and z < self.c.z:
                return [Line(self.a, self.b, np.array([self.normal.x, self.normal.y]))]
            # 如果 z 切割平面通过顶点 B 和 C 并且在 A 之上，则返回 BC 线段
            elif z > self.a.z and z == self.b.z and z == self.c.z:
                return [Line(self.b, self.c, np.array([self.normal.x, self.normal.y]))]
            # 如果 z 切割平面在 A 和 B 之间并且在 C 之下，则返回 AB 和 AC 的交点
            elif self.a.z < z < self.b.z and z < self.c.z:
                # 计算 AB 的交点
                ratio0 = (z - self.a.z) / (self.b.z - self.a.z)
                x0 = ratio0 * (self.b.x - self.a.x) + self.a.x
                y0 = ratio0 * (self.b.y - self.a.y) + self.a.y
                p0 = Point(x=x0, y=y0, z=z)
                # 计算 AC 的交点
                ratio1 = (z - self.a.z) / (self.c.z - self.a.z)
                x1 = ratio1 * (self.c.x - self.a.x) + self.a.x
                y1 = ratio1 * (self.c.y - self.a.y) + self.a.y
                p1 = Point(x=x1, y=y1, z=z)
                return [Line(p0, p1, np.array([self.normal.x, self.normal.y]))]
            # 如果 z 切割平面在 A 和 C 之间并且通过 B，则返回 B 和 AC 的交点
            elif self.a.z < z < self.c.z and z == self.b.z:
                # 计算 AC 的交点
                ratio = (z - self.a.z) / (self.c.z - self.a.z)
                x0 = ratio * (self.c.x - self.a.x) + self.a.x
                y0 = ratio * (self.c.y - self.a.y) + self.a.y
                p0 = Point(x=x0, y=y0, z=z)
                return [Line(p0, self.b, np.array([self.normal.x, self.normal.y]))]
                # return None
            # 如果 z 切割平面在 B 和 C 之间并且在 A 之上，则返回 AC 和 BC 的交点
            elif self.b.z < z < self.c.z and z > self.a.z:
                # 计算 AC 的交点
                ratio0 = (z - self.a.z) / (self.c.z - self.a.z)
                x0 = ratio0 * (self.c.x - self.a.x) + self.a.x
                y0 = ratio0 * (self.c.y - self.a.y) + self.a.y
                p0 = Point(x=x0, y=y0, z=z)
                # 计算 BC 的交点 
                ratio1 = (z - self.b.z) / (self.c.z - self.b.z)
                x1 = ratio1 * (self.c.x - self.b.x) + self.b.x
                y1 = ratio1 * (self.c.y - self.b.y) + self.b.y
                p1 = Point(x=x1, y=y1, z=z)
                return [Line(p0, p1, np.array([self.normal.x, self.normal.y]))]
            # C
            elif z > self.a.z and z > self.b.z and z == self.c.z:
                # return [Line(self.c, self.c, np.array([self.normal.x, self.normal.y]))]
                return None
            # ABC
            elif z == self.a.z == self.b.z == self.c.z:
                return [
                    Line(self.a, self.b, np.array([self.normal.x, self.normal.y])),
                    Line(self.a, self.c, np.array([self.normal.x, self.normal.y])),
                    Line(self.b, self.c, np.array([self.normal.x, self.normal.y])),
                ]


class Model:
    """
    读取三维模型并展现为三角面片格式
    """
    def __init__(self, filename:str) -> None:
        self.triangles = []
        self.triangles_count = -1
        self.min_x = self.max_x = 0
        self.min_y = self.max_y = 0
        self.min_z = self.max_z = 0

        with open(filename, 'rb') as fp:
            buffer = fp.read()
            #print(buffer[:100])
        
        if buffer.startswith("solid".encode("utf-8")):
            self.__read_from_txt(buffer.decode('utf-8'))
        else:
            self.__read_from_binary(buffer)
        
        self.volume = self.cal_volume()
        self.__update_bounds()
        self.move_to_center()

    @property
    def length(self) -> float:
        return self.max_x - self.min_x

    @property
    def width(self) -> float:
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    def __read_from_txt(self, buffer):
        lines = buffer.splitlines()
        for index in range(1, len(lines), 7):
            if index + 6 > len(lines):
                break
            
            # 将各行数据拆分成列表
            normal = lines[index].split(' ')
            point_a = lines[index + 2].split(' ')
            point_b = lines[index + 3].split(' ')
            point_c = lines[index + 4].split(' ')

            triangle = Triangle(
                a=Point(
                    x=float(point_a[-3]), y=float(point_a[-2]), z=float(point_a[-1])
                ),
                b=Point(
                    x=float(point_b[-3]), y=float(point_b[-2]), z=float(point_b[-1])
                ),
                c=Point(
                    x=float(point_c[-3]), y=float(point_c[-2]), z=float(point_c[-1])
                ),
                n=FaceNormal(
                    x=float(normal[-3]), y=float(normal[-2]), z=float(normal[-1])
                ),
            )
            self.triangles.append(triangle)
        
        self.triangles_count = len(self.triangles)

    def __read_from_binary(self, buffer):
        if len(buffer) == 0:
            return
        
        self.triangles_count = struct.unpack('1i', buffer[80:84])[0]
        for index in range(self.triangles_count):
            triangle = Triangle(buffer=buffer[84 + index * 50 : 84 + (index + 1) * 50])
            self.triangles.append(triangle)

    def __update_bounds(self):
        self.min_x = min(node.min_x for node in self.triangles)
        self.max_x = max(node.max_x for node in self.triangles)
        self.min_y = min(node.min_y for node in self.triangles)
        self.max_y = max(node.max_y for node in self.triangles)
        self.min_z = min(node.min_z for node in self.triangles)
        self.max_z = max(node.max_z for node in self.triangles)   
        # print(self.min_x,  self.max_x, self.min_y, self.max_y, self.min_z, self.max_z)    

    def cal_volume(self):
        """
        计算模型体积
        来自于 [1]王泉德.任意三角网格模型体积的快速精确计算方法[J].计算机工程与应用,2009(18):32-34,58.
        :return:
        """
        volume = 0
        vec = np.array([0, 0, 1])
        for triangle in self.triangles:
            d = np.dot(
                np.array([triangle.normal.x, triangle.normal.y, triangle.normal.z]), vec
            )
            if d == 0:
                continue
            vf = np.array(
                [
                    [1, 1, 1, 1],
                    [triangle.a.x, triangle.b.x, triangle.c.x, triangle.b.x],
                    [triangle.a.y, triangle.b.y, triangle.c.y, triangle.b.y],
                    [triangle.a.z, triangle.b.z, triangle.c.z, self.min_z],
                ]
            )
            vs = np.array(
                [
                    [1, 1, 1, 1],
                    [triangle.a.x, triangle.a.x, triangle.b.x, triangle.c.x],
                    [triangle.a.y, triangle.a.y, triangle.b.y, triangle.c.y],
                    [triangle.a.z, self.min_z, self.min_z, triangle.c.z],
                ]
            )
            vt = np.array(
                [
                    [1, 1, 1, 1],
                    [triangle.a.x, triangle.b.x, triangle.c.x, triangle.c.x],
                    [triangle.a.y, triangle.b.y, triangle.c.y, triangle.c.y],
                    [self.min_z, self.min_z, self.min_z, triangle.c.z],
                ]
            )

            v1 = 1 / 6 * abs(np.linalg.det(vf))
            v2 = 1 / 6 * abs(np.linalg.det(vs))
            v3 = 1 / 6 * abs(np.linalg.det(vt))
            vs = v1 + v2 + v3
            if d > 0:
                volume += vs
            if d < 0:
                volume -= vs
        return volume

    def move_to_center(self):
        # 计算模型在 x, y, z 方向上的中心点坐标
        center_x = (self.min_x + self.max_x) / 2
        center_y = (self.min_y + self.max_y) / 2
        center_z = (self.min_z + self.max_z) / 2
        # 创建一个平移矩阵，将模型的中心点移动到原点       
        T = transformations.translation_matrix([-center_x, -center_y, -center_z]).T

        # 遍历所有三角形，应用平移矩阵进行位置变换
        for triangle in self.triangles:
            triangle.transform(T)
        # 重新计算并更新模型的边界值
        self.__update_bounds()        

    def get_slice(self, z) -> list[Line]:
        cross_lines = []
        if z < self.min_z or z > self.max_z:
            pass
        else:
            for index in range(self.triangles_count):
                line = self.triangles[index].slice_z(z)
                if line is None:
                    continue
                else:
                    cross_lines.extend(line)

        # 将 lines 打印到一个文本文件中
        # self.log_lines_to_file(z, cross_lines)
        
        return cross_lines
    
    def log_lines_to_file(self, z, lines):
        file_path = "F:/workdir/modelfeatureV2/data/slices_output_py.txt"
        with open(file_path, 'a') as file:
            # 写入当前z值
            file.write(f"Slice at Z = {z}:\n")
            # 写入所有的line
            for line in lines:
                file.write(str(line) + "\n")
            # 添加分隔符
            file.write("-" * 50 + "\n")

# modelRead = Model('F:/workdir/n2_calculate/data/stlModel/ASHAFT_PRT.STL')


