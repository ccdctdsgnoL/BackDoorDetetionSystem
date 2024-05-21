import torch
import torch.nn as nn


class BaselineMNISTNetwork(nn.Module):
    """
    MNIST数据集的基准网络。

    该网络是基准网络的实现，用于处理MNIST数据集，参考论文
    BadNets: Evaluating Backdooring Attacks on Deep Neural Networks <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8685687&tag=1>
    """

    def __init__(self):
        super(BaselineMNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)  # 卷积层1 输入通道数1，输出
        """
        nn.Conv2d(in_channels, out_channels, kernel_size)
        in_channels: 输入通道的数量，对于灰度图像而言，通道数为 1。对于彩色图像，通常有 3 个通道 (红色、绿色、蓝色)
        out_channels: 输出通道的数量，即卷积核的数量，决定了卷积层学习的特征数
        kernel_size: 卷积核的大小，可以是一个整数（表示正方形卷积核的边长）或一个元组（表示高度和宽度的大小）
        
        nn.Conv2d(1, 16, 5) 表示一个输入通道、输出通道为 16、卷积核大小为 5x5 的卷积层。这个卷积层将输入的图像进行卷积操作，生成 16 个特征图。
        """
        self.conv2 = nn.Conv2d(16, 32, 5)  # 卷积层2
        self.fc1 = nn.Linear(512, 512)  # 全连接层1
        self.fc2 = nn.Linear(512, 10)  # 全连接层2

        self.avg_pool = nn.AvgPool2d(2)  # 池化层
        self.relu = nn.ReLU(inplace=True)  # 激活函数
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 打印输入数据的大小
        # print("输入数据大小: ", x.size())
        x = self.conv1(x)  # 第一层卷积
        # 打印第一层卷积后的数据大小
        # print("第一层卷积后大小: ", x.size())
        x = self.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)  # 第二层卷积
        # 打印第二层卷积后的数据大小
        # print("第二层卷积后大小: ", x.size())
        x = self.relu(x)
        x = self.avg_pool(x)

        x = x.contiguous().view(-1, 512)  # 展平操作以便输入到全连接层
        # 打印展平后的数据大小
        # print("展平后大小: ", x.size())
        
        x = self.fc1(x)  # 第一层全连接
        # 打印第一层全连接后的数据大小
        # print("第一层全连接后大小: ", x.size())
        x = self.relu(x)

        x = self.fc2(x) # 第二层全连接
        # 打印第二层全连接后的数据大小
        # print("第二层全连接后大小: ", x.size())
        # x = self.softmax(x)

        return x

if __name__ == '__main__':
    baseline_MNIST_network = BaselineMNISTNetwork()
    x = torch.randn(16, 1, 28, 28)
    x = baseline_MNIST_network(x)
    print(x.size())
    print(x)
