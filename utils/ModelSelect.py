"""
用来快速切换模型的代码
"""
import torchvision

from torchvision.transforms import Compose, ToTensor
from models import BaselineMNISTNetwork, ResNet
from .TestedModel import TestedModel


def selectModel(model_name, model_path=None):
    """
    model_name: 模型名称
    mnist: MNIST数据集
    fashion_mnist: FashionMNIST数据集
    cifar10: CIFAR10数据集
    """
    transform = Compose([ToTensor()])
    model_name = model_name.lower().replace("_", "")
    if model_name == "mnist":
        model=BaselineMNISTNetwork()
        mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
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
        return TestedModel(model, model_path, 256), mnist_dataset
    elif model_name == "fashionmnist":
        model=BaselineMNISTNetwork()
        fashion_mnist_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
        model_paths = [
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_FashionMNIST_2024-04-24_11-50-19\ckpt_epoch_30.pth",  # 后门标签 0 右下角
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_FashionMNIST_benign_2024-04-24_11-55-00\ckpt_epoch_30.pth"
        ]
        if model_path is None:
            model_path=model_paths[0]
        elif type(model_path) == int:
            model_path=model_paths[model_path]
        return TestedModel(model, model_path, 256), fashion_mnist_dataset
    elif model_name == "cifar10":
        model=ResNet(18)
        cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
        model_paths = [
            "E:\Code\My_python_files\BackdoorBox\experiments\BadNets_ResNet-18_2024-04-01_15-04-40\ckpt_epoch_200.pth"  # 后门标签 1 右下角
        ]
        if model_path is None:
            model_path=model_paths[0]
        elif type(model_path) == int:
            model_path=model_paths[model_path]
        return TestedModel(model, model_path, 64), cifar10_dataset
    else:
        return None