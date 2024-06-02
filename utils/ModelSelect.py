"""
用来快速切换模型的代码
"""
import torchvision
import torch

from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset
from models import BaselineMNISTNetwork, ResNet
from .TestedModel import TestedModel


def selectModel(model_name, model_path=None, custom_model=None, dataset_path=None):
    """
    model_name: 模型名称
    mnist: MNIST数据集
    fashion_mnist: FashionMNIST数据集
    cifar10: CIFAR10数据集
    """
    transform = Compose([ToTensor()])
    model_name = model_name.lower().replace("_", "")
    if model_name == "自动选择":
        if "fashion" in model_path.lower():
            model_name = "fashionmnist"
        elif "mnist" in model_path.lower():
            model_name = "mnist"
        elif "cifar" in model_path.lower():
            model_name = "cifar10"
        else:
            raise ValueError("不在自动选择范围内")
        print("模型名：", model_name)
        print("模型路径：", model_path)
    print("数据集路径：", dataset_path)
    model = None
    dataset = None
    if model_name == "mnist":
        model=BaselineMNISTNetwork()
        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
        model_paths = [
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_MNIST_2024-03-03_22-51-27\ckpt_epoch_30.pth",  # 后门标签 1 右下角
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_MNIST_2024-03-03_22-56-07\ckpt_epoch_30.pth",
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_MNIST_2024-03-06_19-05-52\ckpt_epoch_30.pth",
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_MNIST_2024-03-26_16-31-13\ckpt_epoch_30.pth",
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_MNIST_benign_2024-03-26_16-35-32\ckpt_epoch_30.pth"
        ]
        if model_path is None:
            model_path=model_paths[0]
        elif type(model_path) == int:
            model_path=model_paths[model_path]
    elif model_name == "fashionmnist":
        model=BaselineMNISTNetwork()
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
        model_paths = [
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_FashionMNIST_2024-04-24_11-50-19\ckpt_epoch_30.pth",  # 后门标签 0 右下角
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_FashionMNIST_benign_2024-04-24_11-55-00\ckpt_epoch_30.pth"
        ]
        if model_path is None:
            model_path=model_paths[0]
        elif type(model_path) == int:
            model_path=model_paths[model_path]
    elif model_name == "cifar10":
        model=ResNet(18)
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
        model_paths = [
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_ResNet-18_2024-04-01_15-04-40\ckpt_epoch_200.pth"  # 后门标签 1 右下角
        ]
        if model_path is None:
            model_path=model_paths[0]
        elif type(model_path) == int:
            model_path=model_paths[model_path]
    elif model_name == "自定义模型":
        model = custom_model
        dataset = None
    
    else:
        return None
    if dataset_path == "自动加载":
        return TestedModel(model, model_path, 64), dataset
    elif dataset_path == "MNIST":
        return TestedModel(model, model_path, 256), torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
    elif dataset_path == "FashionMNIST":
        return TestedModel(model, model_path, 256), torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
    elif dataset_path == "CIFAR10":
        return TestedModel(model, model_path, 64), torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    else:
        return TestedModel(model, model_path, 64), torch.load(dataset_path)
    
# 定义一个自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path):
        # 加载.pt文件
        self.data = torch.load(data_path)

    def __len__(self):
        # 返回数据集的大小
        return self.data.size(0)

    def __getitem__(self, index):
        # 返回索引index对应的数据项
        return self.data[index]
    
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']