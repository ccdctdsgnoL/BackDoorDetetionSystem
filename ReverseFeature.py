"""
特征逆向模块
指定标签，寻找对应的特征图像
"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time

from models import BaselineMNISTNetwork
from torchvision.transforms import Compose, ToTensor
from models.baseline_MNIST_network import BaselineMNISTNetwork
from torch.utils.data import DataLoader, Subset
from utils import save_tensor_image

class Generator(nn.Module):
    def __init__(self, input_size, image_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(input_size, image_size * image_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.view(-1, 1, 28, 28)


if __name__ == '__main__':
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = r"D:\Code\My_python_files\BackdoorBox\experiments\BadNets_MNIST_2024-03-03_22-51-27\ckpt_epoch_30.pth"
    dataset = torchvision.datasets.MNIST('data', train=True, transform=Compose([ToTensor()]), download=False)
    # 选择 10000 条数据的子集
    subset_indices = range(10000)
    
    
    model = BaselineMNISTNetwork().to(device)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint)
    model.eval()

    # 定义模型参数
    input_size = 10  # 输入大小，即预测的目标张量的大小
    image_size = 28  # 图像大小，即生成的图像张量的大小

    # 实例化生成器模型，并移动到GPU上
    generator = Generator(input_size, image_size).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([generator.linear.weight, generator.linear.bias], lr=0.01)

    # 后门标签
    backdoor_label = 1

    # 目标张量，并移动到GPU上
    predicted_labels = torch.zeros(1, input_size).to(device)

    # 训练生成器模型
    num_epochs = 100
    start_time = time.time()
    print("Training Generator Model...")
    total_predicted_labels_avg = predicted_labels
    for epoch in range(num_epochs):
        # 清零梯度
        optimizer.zero_grad()
        # 生成后门图片 前向传播
        backdoor_tensor = generator(total_predicted_labels_avg)
        total_predicted_labels_avg.zero_()
        # 初始化总预测标签总和
        total_predicted_labels_sum = torch.zeros(1, 10).to(device)
        
        subset = Subset(dataset, subset_indices)
        # 创建数据加载器
        batch_size = 128
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        # 遍历数据集的每个批次
        total_batches = len(dataloader)
        # 初始化一个长度为10的张量，用于统计每个类别的预测数量
        class_counts = torch.zeros(10)
        for images, real_labels in dataloader:
            # 向图片添加后门，并限制结果不能超过1.0
            # 将输入图像移动到GPU上
            images = images.to(device)
            images += backdoor_tensor
            images = torch.clamp(images, min=0.0, max=1.0)

            with torch.no_grad():
                # 模型预测
                predicted_labels = model(images)
                predicted_labels = predicted_labels.requires_grad_()  # 将梯度开启
            # 累加预测标签
            total_predicted_labels_sum += predicted_labels.sum(dim=0)
            # 找到每个样本中概率最大的类别
            max_prob_indices = torch.argmax(predicted_labels, dim=1)
            # 统计每个类别出现的次数
            class_counts += torch.bincount(max_prob_indices, minlength=10).cpu()
        print("每个类别的预测数量：", class_counts)
        # 计算总平均值
        total_predicted_labels_avg = total_predicted_labels_sum / (total_batches * 128)
        # 计算损失
        loss = criterion(total_predicted_labels_avg, torch.tensor([backdoor_label]).to(device))
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 输出损失
        if (epoch+1) % 10 == 0:
            end_time = time.time()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s')
            # print(total_predicted_labels_avg.cpu().detach().numpy()[0])
            start_time = time.time()
            # show_tensor_image(backdoor_tensor.cpu())
            save_tensor_image(images.cpu(), f"D:\\Code\\My_python_files\\BackdoorBox\\NewTest\\backdoorImages\\{epoch+1}.png")
            # print(generator.linear)