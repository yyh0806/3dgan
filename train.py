import torch
import os
from torch import optim
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from model import InfoGANGeneratorWithMixedCodes, OptimizedInfoGANDiscriminator
from torch.utils.tensorboard import SummaryWriter
import trimesh
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 损失函数定义
def define_loss_functions():
    def discriminator_loss(real_output, fake_output):
        real_loss = torch.nn.functional.binary_cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = torch.nn.functional.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        return (real_loss + fake_loss) / 2

    def generator_loss(fake_output):
        return torch.nn.functional.binary_cross_entropy(fake_output, torch.ones_like(fake_output))

    def info_loss(categorical_pred, continuous_pred, categorical_true, continuous_true):
        categorical_loss = torch.nn.functional.cross_entropy(categorical_pred, categorical_true)
        continuous_loss = torch.nn.functional.mse_loss(continuous_pred, continuous_true)
        return categorical_loss + continuous_loss

    return discriminator_loss, generator_loss, info_loss


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


# 数据加载器
def get_data_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageObjDataset(images_root='./dataset/images', objs_root='./dataset/objs', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练函数
def train(generator, discriminator, data_loader, optimG, optimD, epochs, device):
    writer = SummaryWriter('runs/infogan_experiment')
    disc_loss_fn, gen_loss_fn, info_loss_fn = define_loss_functions()
    
    for epoch in range(epochs):
        for i, (real_images, real_objs) in enumerate(data_loader):
            real_images, real_objs = real_images.to(device), real_objs.to(device)
            noise = torch.randn(real_images.size(0), noise_dim, device=device)
            c_cat = torch.randn(real_images.size(0), num_categories, device=device)
            c_cont = torch.randn(real_images.size(0), cont_dim, device=device)

            # Train Discriminator
            optimD.zero_grad()
            # 这里要确认 real_output到底是什么，后面两个东西有没有用，我给删掉了
            real_output, _, _ = discriminator(real_objs)
            # 这里少一维度，因为你生成器要加图像，所以我加了一个real_images，你要确定对不对
            fake_objs = generator(real_images, noise, c_cat, c_cont)
            # 这里要确认 fake_output到底是什么，后面两个东西有没有用，我给删掉了
            fake_output, _, _ = discriminator(fake_objs.detach())
            d_loss = disc_loss_fn(real_output, fake_output)
            d_loss.backward()
            optimD.step()

            # Train Generator
            optimG.zero_grad()
            fake_output, c_cat_pred, c_cont_pred = discriminator(fake_objs)
            g_loss = gen_loss_fn(fake_output)
            info_l = info_loss_fn(c_cat_pred, c_cont_pred, c_cat, c_cont)
            total_g_loss = g_loss + info_l
            total_g_loss.backward()
            optimG.step()

            # Logging
            if i % 100 == 0:
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(data_loader) + i)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(data_loader) + i)
                writer.add_scalar('Loss/Info', info_l.item(), epoch * len(data_loader) + i)

        print(f"Epoch {epoch+1}, Discriminator Loss: {d_loss.item()}, Generator Loss: {total_g_loss.item()}")

    writer.close()

# 参数设置
noise_dim = 100      # 噪声向量的维度
num_categories = 4  # 类别的数量
cont_dim = 5        # 连续编码的维度
img_channels = 3     # 图像通道数
img_size = 256       # 图像尺寸
point_cloud_dim = 26404*3  # 点云的输出维度
num_samples = 32    # 样本数量

# 主执行函数
def main():
    batch_size = 32
    epochs = 100
    generator = InfoGANGeneratorWithMixedCodes(noise_dim, num_categories, cont_dim, img_channels, img_size, point_cloud_dim).to(device)
    discriminator = OptimizedInfoGANDiscriminator(num_categories, cont_dim).to(device)
    data_loader = get_data_loader(batch_size)
    
    optimG = optim.Adam(generator.parameters(), lr=0.0002)
    optimD = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    train(generator, discriminator, data_loader, optimG, optimD, epochs, device)

    # Save models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

if __name__ == "__main__":
    main()