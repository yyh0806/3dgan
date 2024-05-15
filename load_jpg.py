import torch
from torchvision import transforms
from PIL import Image

# 定义图像转换操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 读取并转换图像
image_path = './dataset/objs/101/4_anger.jpg'
image = Image.open(image_path)
img_tensor = transform(image)

# 检查转换后的张量
print(img_tensor.shape)  # 输出: torch.Size([3, 256, 256])
print(img_tensor.dtype)  # 输出: torch.float32
