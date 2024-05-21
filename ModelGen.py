"""
训练毒化模型和良性模型
随机生成触发器和后门标签
"""


import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
import models

from torchvision.transforms import Compose, ToTensor, ToPILImage, RandomHorizontalFlip


def save_poisoned_datasets(poisoned_train, poisoned_test, save_path):
    # 保存毒化后的训练和测试数据集
    torch.save(poisoned_train, f'{save_path}/poisoned_train.pth')
    torch.save(poisoned_test, f'{save_path}/poisoned_test.pth')

def backDoorStyle(height, width, ext_val=5):
    """
    height: 图像高度
    width: 图像宽度
    ext_val: 扩展值（最大的触发器形状）
    return: 黑白色所有的触发器张量
    """
    # 初始化一个触发器列表
    triggers_list = []
    for e in range(ext_val, 0, -1):  # 遍历扩展值
        for y in range(height-e+1):  # 遍历图像高度
            for x in range(width-e+1):  # 遍历图像宽度
                trigger = torch.zeros(1, height, width)
                trigger[0, y:y+e, x:x+e] = 1 # 填充触发器
                triggers_list.append(trigger)
    triggers = torch.stack(triggers_list)  # 堆叠触发器
    return triggers

def randomChoiceData(dataset, n):
    """
    随机抽取数据
    dataset: 数据集
    n: 选择的样本数量
    return: 选择的样本
    """
    dataset_len = len(dataset)
    while n > dataset_len:
        n = n >> 1
    batch_indices = np.random.choice(dataset_len, n)
    batch_x = torch.stack([dataset[i][0] for i in batch_indices])

    return batch_x

def saveTrigger(weight, t_label, save_path):
    # 保存触发器样式和权重
    to_pil = ToPILImage()
    image = to_pil(weight)
    image.save(f'{save_path}/Trigger_{t_label}.png')
    
def train(model, dataset, experiment_dir, n):
    # 加载数据集
    transform_train = Compose([ToTensor(),RandomHorizontalFlip()])
    trainset = dataset('data', train=True, transform=transform_train, download=True)

    transform_test = Compose([ToTensor()])
    testset = dataset('data', train=False, transform=transform_test, download=True)
    
    all_triggers = backDoorStyle(len(testset.data[0][0]), len(testset.data[0][1]), 5)
    selected_triggers = randomChoiceData(all_triggers, n)
    t_labels = [random.randint(0, 9) for _ in range(n)]  
    for i in range(len(selected_triggers)):
        pattern = (selected_triggers[i]*255).to(dtype=torch.uint8)  # 触发器样式
        weight = selected_triggers[i]  # 触发器权重
        t_label = t_labels[i]
        
        # 使用指定的超参数初始化 BadNets
        badnets = models.BadNets(
            train_dataset=trainset,  # 用户应采用其训练数据集。
            test_dataset=testset,    # 用户应采用其测试数据集。
            model=model,  # 用户可以采用其模型。
            loss=nn.CrossEntropyLoss(),
            y_target=t_label,  # 毒化目标标签
            poisoned_rate=0.05,
            pattern=pattern,
            weight=weight,
            deterministic=False,  # True
            seed=666
        )

        # 获取有毒的训练和测试数据集
        poisoned_train, poisoned_test = badnets.get_poisoned_dataset()

        # 训练并获取受攻击的模型
        schedule = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': '0',
            'GPU_num': 1,
            'benign_training': False,
            'batch_size': 128,  # 128
            'num_workers': 0,
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'schedule': [150, 180],
            'epochs': 100,  # 200
            'log_iteration_interval': 100,
            'test_epoch_interval': 10,
            'save_epoch_interval': 10,
            'save_dir': 'experiments',
            'experiment_name': experiment_dir
        }
        # 训练良性的模型
        schedule_benign = {
            'device': 'GPU',
            'CUDA_VISIBLE_DEVICES': '0',
            'GPU_num': 1,
            'benign_training': True,
            'batch_size': 128,  # 128
            'num_workers': 0,
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'schedule': [150, 180],
            'epochs': 30,  # 200
            'log_iteration_interval': 200,
            'test_epoch_interval': 10,
            'save_epoch_interval': 30,
            'save_dir': 'experiments',
            'experiment_name': experiment_dir+"_benign"
        }

        # 训练并获取受攻击的模型
        badnets.train(schedule)
        saveTrigger(weight, t_label, badnets.work_dir)
        # attacked_model = badnets.get_model()  # 获取受攻击的模型。
        # badnets.train(schedule_benign)
        print(f"已经训练完成：{i}/{len(selected_triggers)}")

if __name__ == '__main__':
    train_data_name = "CIFAR10"
    
    if train_data_name == "MNIST":
        dataset = torchvision.datasets.MNIST  # 使用torchvision加载MNIST数据集
        experiment_dir = "MNIST"
        model = models.BaselineMNISTNetwork()
    elif train_data_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10  # 使用torchvision加载CIFAR10数据集
        experiment_dir = "CIFAR10"
        model = models.ResNet(18)
    elif train_data_name == "FashionMNIST":
        dataset = torchvision.datasets.FashionMNIST  # 使用torchvision加载FashionMNIST数据集
        experiment_dir = "FashionMNIST"
        model = models.BaselineMNISTNetwork()
    else:
        print(train_data_name)
        exit()
    
    train(model, dataset, experiment_dir, 3)