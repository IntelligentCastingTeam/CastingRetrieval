import vtk

def add_axes_to_interactor(render_window_interactor):
    """
    在渲染窗口的交互器中添加坐标轴
    """
    # 创建坐标轴
    axes = vtk.vtkAxesActor()
    
    # 创建窗口部件，用于显示坐标轴
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(render_window_interactor)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)  # 左下角的20%大小窗口
    widget.SetEnabled(1)
    widget.InteractiveOn()
    
    return widget  # 返回 widget 以确保它在后续使用时未被销毁

def main():
    # 创建一个简单的立方体模型
    cube = vtk.vtkCubeSource()

    # 创建Mapper和Actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # 创建Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # 设置背景颜色

    # 创建RenderWindow
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("VTK with Axes")
    render_window.SetSize(600, 600)
    render_window.AddRenderer(renderer)

    # 创建RenderWindowInteractor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # 设置交互样式
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # 添加坐标轴
    widget = add_axes_to_interactor(interactor)

    # 初始化交互器并启动渲染
    interactor.Initialize()
    render_window.Render()
    
    # 启动交互器
    interactor.Start()

if __name__ == "__main__":
    main()