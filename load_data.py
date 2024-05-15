import trimesh

# 读取 .obj 文件
mesh = trimesh.load('./dataset/objs/101/1_neutral.obj')

# 打印顶点数据
print(mesh.vertices)

# 打印面数据
print(mesh.faces)

# 可视化模型
mesh.show()
