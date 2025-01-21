import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

fileVoxel = "F:/workdir/n2_calculate/data/voxelModel/ErZhouBi.npy"
fileFillInside = "F:/workdir/n2_calculate/data/fill_inside/ErZhouBi.npy"
fileCavities = "F:/workdir/n2_calculate/data/labeled_cavities/ErZhouBi.npy"
file2 = "F:/workdir/n2_calculate/data/labeled_cavities/6531ec72c059807382725e66.npy"
fileTest01 = "F:/workdir/modelfeatureV2/data/fillInside/1012A_.npy"
fileTest02 = "F:/workdir/n2_calculate/data/fill_inside/1012A_.npy"
fileTest03 = "F:/workdir/modelfeatureV2/data/voxelShowTest/test_01.npy"


model = np.load(fileTest01).astype(
    np.uint8
)
other = np.zeros_like(model)

values = list(set(model.flatten()))
print("values:", values)
if 0 not in values:
    for index in range(len(values)):
        other[model == values[index]] = index + 1
else:
    for index in range(len(values)):
        other[model == values[index]] = index
other = other / len(values) * 255

shape = model.shape
flat_data_array = model.flatten()
vtk_data = numpy_to_vtk(num_array=other.ravel(), array_type=vtk.VTK_UNSIGNED_INT)

img = vtk.vtkImageData()
img.SetDimensions(shape[2], shape[1], shape[0])
img.GetPointData().SetScalars(vtk_data)

mapper = vtk.vtkGPUVolumeRayCastMapper()
mapper.SetInputData(img)
mapper.Update()

opacity = vtk.vtkPiecewiseFunction()
opacity.AddPoint(0, 0)
opacity.AddPoint(255, 0.8)

color = vtk.vtkColorTransferFunction()
color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
color.AddRGBPoint(255.0, 0.8, 0.8, 0.8)

volume_property = vtk.vtkVolumeProperty()
volume_property.SetScalarOpacity(opacity)
volume_property.SetColor(color)
volume_property.SetInterpolationTypeToNearest()
volume_property.ShadeOff()
# 设置环境光系数。环境光系数为 1 表示体素将完全受环境光影响,所有部分都同样明亮
volume_property.SetAmbient(1)
# 漫反射光系数
volume_property.SetDiffuse(0.5)
# 高光反射光系数
volume_property.SetSpecular(0.3)


volume = vtk.vtkVolume()
volume.SetMapper(mapper)
volume.SetProperty(volume_property)

renderer = vtk.vtkRenderer()
# (0, 0)：视口的左下角坐标（相对于窗口的归一化坐标）
# (1, 1)：视口的右上角坐标（相对于窗口的归一化坐标）
renderer.SetViewport(0, 0, 1, 1)
# 渲染器的背景颜色将是白色
renderer.SetBackground(1, 1, 1)
renderer.AddVolume(volume)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800, 600)
window.SetWindowName("Visualization Test")

# 一种交互样式，它允许用户像使用轨迹球一样旋转、平移和缩放摄像机视图。
style = vtk.vtkInteractorStyleTrackballCamera()
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(style)
interactor.SetRenderWindow(window)

window.Render()
interactor.Initialize()
interactor.Start()


