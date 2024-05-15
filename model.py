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
num_samples = 32    # 样本数量

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
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),  
            nn.Linear(512 * (img_size // 16) * (img_size // 16), self.img_feature_dim), 
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
   

class OptimizedInfoGANDiscriminator(nn.Module):
    def __init__(self, num_categories, cont_dim, num_features=26404):
        super(OptimizedInfoGANDiscriminator, self).__init__()
        self.num_features = num_features 
        self.num_categories = num_categories
        self.cont_dim = cont_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        # 判别器的最终判断层,输出真假评分
        self.fc_real_fake = nn.Linear(32 * num_features, 1)
        self.sigmoid = nn.Sigmoid()

        # 类别预测
        self.fc_category = nn.Linear(32 * num_features, num_categories)
        self.softmax = nn.Softmax(dim=1)
        # 连续变量预测
        self.fc_cont = nn.Linear(32 * num_features, cont_dim)
    
    def forward(self, x):
        print("输入向量的形状:", x.shape)
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # 调整输入维度为 [batch_size, 3, 26404]
        print("经过permute操作后的形状:", x.shape)
        x = self.conv_layers(x)
        print("经过卷积以后的形状:", x.shape)
        x = x.view(batch_size, -1)
        real_fake = self.sigmoid(self.fc_real_fake(x))
        category = self.softmax(self.fc_category(x))
        cont_vars = self.fc_cont(x)
        return real_fake, category, cont_vars
    

#写一个例子测试上面的模型是否正常运行
if __name__ == '__main__':
    # 定义模型
    model = InfoGANGeneratorWithMixedCodes(noise_dim, num_categories, cont_dim, img_channels, img_size, point_cloud_dim)
    # 定义输入    
    noise = torch.randn(num_samples, noise_dim)
    c_cat_indices = torch.randint(0, num_categories, (num_samples,))
    c_cat = torch.nn.functional.one_hot(c_cat_indices, num_classes=num_categories).float()
    c_cont = torch.randn(num_samples, cont_dim)
    img = torch.randn(num_samples, img_channels, img_size,img_size)
    # 打印形状
    print(f"noise shape: {noise.shape}")
    print(f"c_cat shape: {c_cat.shape}")
    print(f"c_cont shape: {c_cont.shape}")
    print(f"img shape: {img.shape}")
    # 连接后的输出形状
    output = torch.cat([noise, c_cat, c_cont], dim=1)
    print(f"output shape: {output.shape}")
    # 前向传播
    point_cloud = model(img, noise, c_cat, c_cont)
    print(point_cloud.shape)  # 输出点云的形状

 ######################################################################################################

    discriminator = OptimizedInfoGANDiscriminator(num_categories, cont_dim)
    # 打印形状
    print(f"point_cloud shape: {point_cloud.shape}")
    # 前向传播
    real_fake, category, cont_vars = discriminator(point_cloud)
    print(f"real_fake shape: {real_fake.shape}")
    print(f"category shape: {category.shape}")
    print(f"cont_vars shape: {cont_vars.shape}")