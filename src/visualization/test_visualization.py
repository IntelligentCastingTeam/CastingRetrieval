import os
import tkinter as tk

import cv2
import numpy as np
import open3d as o3d
import pytest
import vtk
from tqdm import tqdm
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy


# 创建一个 30x30x30 的三维数组，并初始化为零
model = np.zeros([30, 30, 30])
# 在数组中指定的区域内设置为 1
model[10:20, 10:12, 14:20] = 1
model[14:20, 10:20, 10:12] = 1


shape = model.shape
data_type = vtk.VTK_FLOAT
flat_data_array = model.flatten()
vtk_data = numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)

# 创建 VTK 图像数据对象
img = vtk.vtkImageData()
img.SetDimensions(shape[0], shape[1], shape[2])
img.GetPointData().SetScalars(vtk_data)

# 创建一个颜色查找表
color_table = vtk.vtkLookupTable()
color_table.SetRange(0, 1)  # 设置数据范围
color_table.SetValueRange(0.0, 1.0)  # 设置颜色范围
color_table.Build()


# 创建一个颜色转移函数
color_function = vtk.vtkColorTransferFunction()
color_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
color_function.AddRGBPoint(1.0, 1.0, 1.0, 1.0)

# 创建一个不透明度转移函数
opacity_function = vtk.vtkPiecewiseFunction()
opacity_function.AddPoint(0, 0.0)
opacity_function.AddPoint(1, 1.0)

# 创建一个体积属性对象
volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(color_function)
volume_property.SetScalarOpacity(opacity_function)
volume_property.SetScalarOpacityUnitDistance(0.1)

# 创建一个映射器
mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputData(img)
mapper.SetBlendModeToComposite()

# 创建一个体积对象
volume = vtk.vtkVolume()
volume.SetMapper(mapper)
volume.SetProperty(volume_property)

# 创建一个渲染器
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume)
renderer.SetBackground(0.1, 0.2, 0.4)  # 设置背景颜色

# 创建一个渲染窗口
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(600, 600)

# 创建一个交互渲染器
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# 开始渲染循环
render_window.Render()
render_window_interactor.Initialize()
render_window_interactor.Start()