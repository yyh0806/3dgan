import trimesh

# 读取 .obj 文件
mesh = trimesh.load('./dataset/objs/101/4_anger.obj')

# 打印顶点数据
print(type(mesh.vertices))

# 打印面数据
print(mesh.faces)

# 可视化模型
mesh.show()
