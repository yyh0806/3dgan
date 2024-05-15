import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import trimesh

class ImageObjDataset(Dataset):
    def __init__(self, images_root, objs_root, transform=None):
        self.images_root = images_root
        self.objs_root = objs_root
        self.transform = transform
        self.image_paths = self._get_all_image_paths(images_root)

    def _get_all_image_paths(self, dir):
        image_paths = []
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _get_obj_path(self, img_path):
        # 获取子文件夹名和文件名
        folder_name = os.path.basename(os.path.dirname(img_path))
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        obj_path = os.path.join(self.objs_root, folder_name, f"{file_name}.obj")
        return obj_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        obj_path = self._get_obj_path(img_path)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"{obj_path} does not exist")
        mesh = trimesh.load(obj_path)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        vertices = torch.from_numpy(vertices)

        return image, vertices

# 定义图像转换操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 创建数据集对象
dataset = ImageObjDataset(images_root='./dataset/images', objs_root='./dataset/objs', transform=transform)

# 使用数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 示例：迭代数据
for images, objs in dataloader:
    print(images.shape)  # 输出: torch.Size([32, 3, 256, 256])
    print(objs.shape)    # 输出: (32, n, 3)，n是每个.obj文件中的顶点数量
