import vtk
from model import O3dModel
from pose_normalize import CPCA_normalize, CPCA_normalize_Guo
from vtkmodules.util.numpy_support import numpy_to_vtk

def convert_o3d_to_vtk(o3d_model:O3dModel):
    """
    将 Open3D 模型转换为 VTK 的 PolyData
    """
    vtk_points = vtk.vtkPoints()
    vtk_triangles = vtk.vtkCellArray()

    vertices = o3d_model.points # 顶点数组
    triangles = o3d_model.triangles # 三角形索引数组

    # 将Open3D模型中的顶点添加到VTK点集中
    for v in vertices:
        vtk_points.InsertNextPoint(v[0], v[1], v[2])

    # 将Open3D模型中的三角形面片添加到VTK的三角形数据结构中
    for triangle in triangles:
        vtk_triangle = vtk.vtkTriangle()
        vtk_triangle.GetPointIds().SetId(0, int(triangle[0]))
        vtk_triangle.GetPointIds().SetId(1, int(triangle[1]))
        vtk_triangle.GetPointIds().SetId(2, int(triangle[2]))
        vtk_triangles.InsertNextCell(vtk_triangle)

    poly_data = vtk.vtkPolyData()       # 创建一个新的PolyData对象
    poly_data.SetPoints(vtk_points)     # 设置模型的顶点
    poly_data.SetPolys(vtk_triangles)   # 设置模型的三角形面片

    # 打印模型的坐标范围
    bounds = poly_data.GetBounds()
    print(f"Model Bounds: {bounds}")

    return poly_data

def create_model_renderer(poly_data: vtk.vtkPolyData, color):
    """
    创建VTK渲染器
    """
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    # 重置相机视角以适应模型
    renderer.ResetCamera()

    return renderer

def create_origin_renderer():
    """
    创建只渲染原点标记(球体)的VTK渲染器
    """
    # 创建一个球体来标记原点
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(0.0, 0.0, 0.0)  # 设置球体的中心为原点
    sphere_source.SetRadius(8)  # 设置球体的半径，根据模型大小调整

    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # 球体颜色为黑色
    sphere_actor.GetProperty().SetOpacity(0.8)  # 设置透明度，避免遮挡模型

    # 创建一个新的渲染器并添加球体
    renderer = vtk.vtkRenderer()
    renderer.AddActor(sphere_actor)
    renderer.SetBackground(0, 0, 0)
    
    # # 确保相机重置                  
    # renderer.ResetCamera()
    
    return renderer

class MouseInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    自定义交互类，用于捕捉鼠标点击事件
    """
    def __init__(self, picker, renderer):
        super(MouseInteractorStyle, self).__init__()
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.picker = picker
        self.renderer = renderer

        # 用于标记拾取点的actor
        self.marker_actor = None

    def left_button_press_event(self, obj, event):
        # 获取点击的屏幕坐标
        click_pos = self.GetInteractor().GetEventPosition()

        # 使用拾取器在屏幕点击位置拾取模型
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        # 获取拾取的 3D 点坐标
        picked_pos = self.picker.GetPickPosition()

        # 获取拾取到的单元ID（即三角形ID）
        cell_id = self.picker.GetCellId()

        if cell_id != -1:
            print(f"Picked point coordinates: {picked_pos}")  # 打印拾取点的3D坐标
            print(f"Picked cell ID: {cell_id}")  # 打印拾取到的三角形ID

            # 添加或更新红色标记点
            self.add_or_update_marker(picked_pos)
        else:
            print("No valid point picked")

        # 保持默认的左键行为（如旋转）
        self.OnLeftButtonDown()
    
    def add_or_update_marker(self, position):
        """
        在拾取点处添加或更新红色的标记点（小球体）。        
        """
        if self.marker_actor is None:
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetRadius(6.0)

            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

            self.marker_actor = vtk.vtkActor()
            self.marker_actor.SetMapper(sphere_mapper)
            self.marker_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色标记
            self.marker_actor.GetProperty().SetOpacity(1.0)      
            
            # 将 actor 添加到渲染器中
            self.renderer.AddActor(self.marker_actor)
        
        # 更新标记点的位置
        self.marker_actor.SetPosition(position)
        # 重新渲染场景以显示新的标记
        self.GetInteractor().GetRenderWindow().Render()



def show_two_models(model_before: vtk.vtkPolyData, model_after: vtk.vtkPolyData):
    """
    展示两个窗口：一个展示原始模型，一个展示一致化处理后的模型
    """
    # 创建模型渲染器（底层）
    renderer_before_model = create_model_renderer(model_before, (1.0, 0.0, 0.0))  # 红色模型
    renderer_after_model = create_model_renderer(model_after, (0.0, 1.0, 0.0))   # 绿色模型

    # 创建原点标记渲染器（顶层）
    renderer_before_origin = create_origin_renderer()
    renderer_after_origin = create_origin_renderer()

    # 创建两个渲染窗口
    render_window_before = vtk.vtkRenderWindow()
    render_window_before.SetWindowName("模型一致化处理前")
    render_window_before.SetSize(600, 600)
    render_window_before.AddRenderer(renderer_before_model)
    render_window_before.AddRenderer(renderer_before_origin)

    render_window_after = vtk.vtkRenderWindow()
    render_window_after.SetWindowName("模型一致化处理后")
    render_window_after.SetSize(600, 600)
    render_window_after.AddRenderer(renderer_after_model)
    render_window_after.AddRenderer(renderer_after_origin)

    # 设置窗口中渲染器层次
    renderer_before_origin.SetLayer(1)
    renderer_before_model.SetLayer(0)
    render_window_before.SetNumberOfLayers(2)

    renderer_after_origin.SetLayer(1)
    renderer_after_model.SetLayer(0)
    render_window_after.SetNumberOfLayers(2)

    # 禁用顶层渲染器的交互，确保交互只在模型层进行
    renderer_before_origin.InteractiveOff()
    renderer_after_origin.InteractiveOff()

    # 重要！！！
    # 使多个渲染器同时共享一个相机，否则原点的渲染效果交互后始终不变！
    camera_before = renderer_before_model.GetActiveCamera()
    renderer_before_origin.SetActiveCamera(camera_before)

    camera_after = renderer_after_model.GetActiveCamera()
    renderer_after_origin.SetActiveCamera(camera_after)

    # 创建拾取器
    picker_before = vtk.vtkCellPicker()
    picker_before.SetTolerance(0.0005)
    picker_after = vtk.vtkCellPicker()
    picker_after.SetTolerance(0.0005)

    # 创建两个窗口的交互器
    style_before = MouseInteractorStyle(picker_before, renderer_before_model)
    interactor_before = vtk.vtkRenderWindowInteractor()
    interactor_before.SetInteractorStyle(style_before)
    interactor_before.SetRenderWindow(render_window_before)

    style_after = MouseInteractorStyle(picker_after, renderer_after_model)
    interactor_after = vtk.vtkRenderWindowInteractor()
    interactor_after.SetInteractorStyle(style_after)
    interactor_after.SetRenderWindow(render_window_after)

    # 在每个渲染窗口中添加坐标轴
    widget_before = add_axes_to_renderer(interactor_before)
    widget_after = add_axes_to_renderer(interactor_after)


    # 初始化交互器并启动渲染
    interactor_before.Initialize()
    interactor_after.Initialize()
    render_window_before.Render()
    render_window_after.Render()

    # 启动交互器
    interactor_before.Start()
    interactor_after.Start()

def add_axes_to_renderer(render_window_interactor):
    """
    在渲染器中添加坐标轴
    """
    axes = vtk.vtkAxesActor()

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(render_window_interactor)
    widget.SetViewport(0.0, 0.0, 0.4, 0.4)
    widget.SetEnabled(1)    # 启用 widget
    widget.InteractiveOn()  # 允许用户拖动该窗口

    return widget # 重要！！！一定要返回，否则 Python 的垃圾回收机制会销毁 widget 对象！

def main(stl_file_path):
    # 读取STL文件并创建O3dModel对象
    model = O3dModel(stl_file_path)

    # 将原始模型转换为VTK格式
    vtk_model_before = convert_o3d_to_vtk(model)

    # 对模型进行CPCA一致化处理
    CPCA_normalize(model)

    # 将处理后的模型转换为VTK格式
    vtk_model_after = convert_o3d_to_vtk(model)
    
    # 可视化展示两个模型
    show_two_models(vtk_model_before, vtk_model_after)

if __name__ == "__main__":
    #stl_file_path = "F:/workdir/n2_calculate/data/stlModel/1411731.STL"
    #stl_file_path = "F:/workdir/modelfeatureV2/data/STLModel/qiaoke01.stl"
    stl_file_path = "G:/qiaoke01.stl"
    
    main(stl_file_path)