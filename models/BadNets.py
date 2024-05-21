'''
这是 BadNets [1] 的实现。

参考文献：
[1] Badnets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access 2019.

'''

import copy
import random

import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose

from .base import *


class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img):
        """
        向图像添加水印触发器。
        参数: img (torch.Tensor): 形状为 (C, H, W) 的张量。

        返回: torch.Tensor: 毒害图像，形状为 (C, H, W)。
        """
        return (self.weight * img + self.res).type(torch.uint8)



class AddDatasetFolderTrigger(AddTrigger):
    """
    向 DatasetFolder 图像添加水印触发器。
    参数:
        pattern (torch.Tensor): 形状为 (C, H, W) 或 (H, W) 的张量。
        weight (torch.Tensor): 形状为 (C, H, W) 或 (H, W) 的张量。
    """


    def __init__(self, pattern, weight):
        super(AddDatasetFolderTrigger, self).__init__()

        if pattern is None:
            raise ValueError("Pattern can not be None.")
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            raise ValueError("Weight can not be None.")
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        # 加速计算
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        """
        获取毒化图像。
        参数:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): 如果 img 是 numpy.ndarray 或 torch.Tensor，则其形状应为 (H, W, C) 或 (H, W)。
        返回:
            torch.Tensor: 毒害图像。
        """

        def add_trigger(img):
            # 添加触发器
            if img.dim() == 2:
                img = img.unsqueeze(0)
                img = self.add_trigger(img)
                img = img.squeeze()
            else:
                img = self.add_trigger(img)
            return img

        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = add_trigger(img)
            # 1 x H x W
            if img.size(0) == 1:
                img = Image.fromarray(img.squeeze().numpy(), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = Image.fromarray(img.permute(1, 2, 0).numpy())
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = add_trigger(img)
                img = img.numpy()
            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0).numpy()
            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = add_trigger(img)
            # H x W x C
            else:
                img = img.permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))


class AddMNISTTrigger(AddTrigger):
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
        weight (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """

    def __init__(self, pattern, weight):
        super(AddMNISTTrigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            self.pattern[0, -2, -2] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 28, 28), dtype=torch.float32)
            self.weight[0, -2, -2] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        # print(img)
        return img


class AddCIFAR10Trigger(AddTrigger):
    """Add watermarked trigger to CIFAR10 image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """

    def __init__(self, pattern, weight):
        super(AddCIFAR10Trigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            self.pattern[0, -3:, -3:] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            self.weight[0, -3:, -3:] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class PoisonedDatasetFolder(DatasetFolder):
    """
    毒化文件夹数据集
    DatasetFolder是 PyTorch 中的一个数据集类，用于处理文件夹结构的数据集。它允许用户按照文件夹结构组织数据，并将每个文件夹作为一个类别。通常，每个文件夹包含属于同一类别的一组样本。
    """
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        # 给图片添加触发器
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, weight))

        # Modify labels
        # 修改标签
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        if y_target != None:
            self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target


class PoisonedMNIST(MNIST):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # 在图片中添加触发器
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddMNISTTrigger(pattern, weight))

        # 修改标签
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        if y_target != None:
            self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(pattern, weight))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        if y_target != None:
            self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


def CreatePoisonedDataset(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index):
    """
    创建毒化数据集
    """
    class_name = type(benign_dataset)  # 数据集类型
    if class_name == DatasetFolder:  # 文件夹
        return PoisonedDatasetFolder(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == MNIST or class_name == FashionMNIST:  # MNIST数据集
        # print("创建毒化MNIST数据集...")
        return PoisonedMNIST(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == CIFAR10:  # CIFAR10数据集
        return PoisonedCIFAR10(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    else:
        raise NotImplementedError  # 不支持的类型


class BadNets(Base):
    """
    使用 BadNets 方法构建毒化数据集。

    参数:
        train_dataset (support_list 中的类型): 良性训练数据集。
        test_dataset (support_list 中的类型): 良性测试数据集。
        model (torch.nn.Module): 网络模型。
        loss (torch.nn.Module): 损失函数。
        y_target (int): N-to-1 攻击目标标签。
        poisoned_rate (float): 毒害样本的比例。
        pattern (None | torch.Tensor): 触发模式，形状为 (C, H, W) 或 (H, W)。
        weight (None | torch.Tensor): 触发模式权重，形状为 (C, H, W) 或 (H, W)。
        poisoned_transform_train_index (int): 将毒害变换插入训练数据集的位置索引。默认值: 0。
        poisoned_transform_test_index (int): 将毒害变换插入测试数据集的位置索引。默认值: 0。
        poisoned_target_transform_index (int): 将毒害目标变换插入的位置索引。默认值: 0。
        schedule (dict): 训练或测试计划。默认值: None。
        seed (int): 全局随机数种子。默认值: 0。
        deterministic (bool): 设置 PyTorch 操作是否必须使用“确定性”算法。
            即，对于相同的输入，在相同的软件和硬件上运行时，始终产生相同的输出。启用时，操作将使用确定性算法（如果可用），
            如果只有非确定性算法可用，则调用时会引发 RuntimeError。默认值: False。
            
        在创建 BadNets 时，会自动创建毒化数据集
    """

    def __init__(self,
                 train_dataset,  # 良性训练数据集
                 test_dataset,  # 良性测试数据集
                 model,  # 网络模型
                 loss,  # 损失函数
                 y_target=None,  # N-to-1 攻击目标标签
                 poisoned_rate=0,  # 毒害样本的比例
                 pattern=None,  # 触发模式，形状为 (C, H, W) 或 (H, W)
                 weight=None,  # 触发模式权重，形状为 (C, H, W) 或 (H, W)。
                 poisoned_transform_train_index=0,  # 将毒化变换插入训练数据集的位置索
                 poisoned_transform_test_index=0,  # 将毒化变换插入测试数据集的位置索引
                 poisoned_target_transform_index=0,  # 将毒化目标变换插入的位置索引
                 schedule=None,  # 训练或测试计划
                 seed=0,  # 全局随机数种子
                 deterministic=False  # 是否使用确定性算法
                 ):
        assert pattern is None or (isinstance(pattern, torch.Tensor) and ((0 < pattern) & (pattern < 1)).sum() == 0), 'pattern should be None or 0-1 torch.Tensor.'

        super(BadNets, self).__init__(
            train_dataset=train_dataset,  # 良性训练数据集
            test_dataset=test_dataset,  # 良性测试数据集
            model=model,  # 网络模型
            loss=loss,  # 损失函数
            schedule=schedule,  # 训练或测试计划
            seed=seed,  # 全局随机数种子
            deterministic=deterministic  # 是否使用确定性算法
        )

        self.poisoned_train_dataset = CreatePoisonedDataset(
            train_dataset,
            y_target,
            poisoned_rate,
            pattern,
            weight,
            poisoned_transform_train_index,
            poisoned_target_transform_index
        )  # 毒化训练数据集

        self.poisoned_test_dataset = CreatePoisonedDataset(
            test_dataset,
            y_target,
            1.0,
            pattern,
            weight,
            poisoned_transform_test_index,
            poisoned_target_transform_index
        )  # 毒化测试数据集
