import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# 参数设置
noise_dim = 100      # 噪声向量的维度
num_categories = 4  # 类别的数量
cont_dim = 5        # 连续编码的维度
img_channels = 3     # 图像通道数
img_size = 256       # 图像尺寸
point_cloud_dim = 26404*3  # 点云的输出维度
num_samples = 1024

class InfoGANGeneratorWithMixedCodes(nn.Module):
   def __init__(self, noise_dim, num_categories, cont_dim, img_channels,img_size, point_cloud_dim):
        super( InfoGANGeneratorWithMixedCodes, self).__init__()
        self.noise_dim = noise_dim
        self.img_feature_dim = 512  # 设定图像特征维度
        self.num_categories = num_categories
        self.cont_dim = cont_dim
         # 首先，将输入的噪声、类别标签和连续变量联合起来
        self.fc_noise_cat_cont = nn.Linear(noise_dim + num_categories + cont_dim, self.img_feature_dim)
        # 编码器部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (128, 128, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (64, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (32, 32, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 输出尺寸: (16, 16, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),  # 扁平化
            nn.Linear(512 * (img_size // 16) * (img_size // 16), self.img_feature_dim),  # 转为特征向量
            nn.ReLU()
        )
        
    # 合并图像特征和噪声编码的全连接层
        self.fc_combined = nn.Linear(self.img_feature_dim * 2, point_cloud_dim)

   def forward(self, img, noise, c_cat, c_cont):
        # 处理噪声和编码
        combined_noise_cat_cont = torch.cat([noise, c_cat, c_cont], dim=1)
        transformed_noise = self.fc_noise_cat_cont(combined_noise_cat_cont)
        # 处理图像
        img_features = self.conv_layers(img)
        # 合并处理后的图像特征和噪声编码
        combined_features = torch.cat([img_features, transformed_noise], dim=1)
        point_cloud = self.fc_combined(combined_features)
        return point_cloud.view(-1, 26404, 3)  # 重整输出为所需的点云形状
   

#写一个例子测试上面的模型是否正常运行
if __name__ == '__main__':
    # 定义模型
    model = InfoGANGeneratorWithMixedCodes(noise_dim, num_categories, cont_dim, img_channels, img_size, point_cloud_dim)
    # 定义输入    
    noise = torch.randn(num_samples, noise_dim)
    c_cat = torch.randint(0, num_categories, (num_samples, 1))
    c_cont = torch.randn(num_samples, cont_dim)
    img = torch.randn(num_samples, img_channels, img_size)
    # 前向传播
    point_cloud = model(img, noise, c_cat, c_cont)
    print(point_cloud.shape)  # 输出点云的形状