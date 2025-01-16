import vtk

# 创建一个立方体
cube = vtk.vtkCubeSource()
cube.Update()

# 创建一个映射器
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(cube.GetOutputPort())

# 创建一个actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 创建一个渲染器
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0, 0, 0)

# 创建一个渲染窗口
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(600, 600)

# 创建一个交互渲染器
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)
style = vtk.vtkInteractorStyleTrackballCamera()
render_window_interactor.SetInteractorStyle(style)

# 开始渲染循环
render_window.Render()
render_window_interactor.Start()