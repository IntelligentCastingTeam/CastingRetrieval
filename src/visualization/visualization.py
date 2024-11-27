import vtk
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk

class VolumeRenderWindow:
    def __init__(self) -> None:
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetViewport(0, 0, 1, 1)
        self.renderer.SetBackground(1, 1, 1)

        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(600, 300)
        self.window.AddRenderer(self.renderer)
        self.window.SetWindowName("Volume Render Window")

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(style)
        self.interactor.SetRenderWindow(self.window)

        self._volume_datas = []

    @ property
    def volume_datas(self) -> list[vtk.vtkImageData]:
        return self._volume_datas
    
    def show(self):
        self.window.Render()
        self.interactor.Initialize()
        self.interactor.Start()

    def add_volume(self, volume, max=255):
        if isinstance(volume, np.ndarray):
            shape = volume.shape
            data_type = vtk.VTK_FLOAT
            flat_data_array = volume.flatten()
            vtk_data = numpy_to_vtk(
                num_array=flat_data_array, deep=True, array_type=data_type
            )
            volume = vtk.vtkImageData()
            volume.GetPointData().SetScalars(vtk_data)
            volume.SetDimensions(shape[0], shape[1], shape[2])
        
        self.renderer.AddVolume(volume)
        self.renderer.ResetCamera()

    def _gen_volume_property(self, max=255) -> vtk.vtkVolumeProperty:
        opacity = vtk.vtkPiecewiseFunction()  # 创建不透明度传递函数
        opacity.AddSegment(0, 0, max, 1)  # 添加不透明度段

        color = vtk.vtkColorTransferFunction()  # 创建颜色传递函数
        color.AddRGBSegment(0, 0, 0, 0, max, 1, 0, 0)  # 添加颜色段

        volume_property = vtk.vtkVolumeProperty()  # 创建体属性对象
        volume_property.SetScalarOpacity(opacity)  # 设置不透明度
        volume_property.SetColor(color)  # 设置颜色
        volume_property.SetInterpolationTypeToLinear()  # 设置插值类型
        volume_property.ShadeOn()  # 打开着色
        volume_property.SetAmbient(1)  # 设置环境光
        volume_property.SetDiffuse(0.8)  # 设置漫反射
        volume_property.SetSpecular(0.2)  # 设置镜面反射

        return volume_property
    
    def clean_volumes(self):
        for volume in self.renderer.GetVolumes():
            self.renderer.RemoveVolume(volume)
        self.volume_datas.clear()
        pass

    def set_volume(self, volume):
        """clean other volumes and add this volume

        Args:
            volume (np.ndarray | vtk.vtkImageData | vtk.vtkVolume): the volume model
        """

        self.clean_volume()
        self.add_volume(volume)
        pass