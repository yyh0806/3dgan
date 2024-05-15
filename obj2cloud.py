import trimesh
import open3d as o3d
import numpy as np

# 读取 .obj 文件
mesh = trimesh.load('./dataset/objs/101/4_anger.obj')

# 获取顶点数据
vertices = np.array(mesh.vertices)

# 创建 open3d 点云对象
point_cloud = o3d.geometry.PointCloud()

# 将顶点数据赋值给点云对象
point_cloud.points = o3d.utility.Vector3dVector(vertices)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud])
