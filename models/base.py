from copy import deepcopy
import math
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10, FashionMNIST

from .log import Log


support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10,
    FashionMNIST
)


def check(dataset):
    return isinstance(dataset, support_list)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Base(object):
    """
    用于后门训练和测试的基础类。

    参数:
        train_dataset (支持列表中的类型): 良性训练数据集。
        test_dataset (支持列表中的类型): 良性测试数据集。
        model (torch.nn.Module): 网络模型。
        loss (torch.nn.Module): 损失函数。
        schedule (字典): 训练或测试的全局调度。默认值: None。
        seed (int): 随机数的全局种子。默认值: 0。
        deterministic (bool): 设置是否使用“确定性”算法进行PyTorch操作。
            即，给定相同的输入，并在相同的软件和硬件上运行时，总是产生相同的输出。
            启用时，操作将在可用时使用确定性算法，如果只有非确定性算法可用，则在调用时引发RuntimeError。默认值: False。
    """


    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None, seed=0, deterministic=False):
        assert isinstance(train_dataset, support_list), 'train_dataset 是一个不受支持的数据集类型，train_dataset 应该是我们支持列表的子类。'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset, support_list), 'test_dataset 是一个不受支持的数据集类型，test_dataset 应该是我们支持列表的子类。'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.work_dir = None
        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None
        self._set_seed(seed, deterministic)

    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            # torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # 提示：在某些版本的CUDA中，循环神经网络（RNNs）和长短时记忆网络（LSTM）可能具有非确定性的行为。
            # 如果要使它们确定性，请参阅 torch.nn.RNN() 和 torch.nn.LSTM() 以获取详细信息和解决方法。


    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_model(self):
        # 获取网络模型
        return self.model

    def get_poisoned_dataset(self):
        # 获取被毒化后的数据集
        # 毒化操作并不在这个 base.py 代码中实现
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch, step, len_epoch):
        factor = (torch.tensor(self.current_schedule['schedule']) <= epoch).sum()

        lr = self.current_schedule['lr']*(self.current_schedule['gamma']**factor)

        """Warmup"""
        if 'warmup_epoch' in self.current_schedule and epoch < self.current_schedule['warmup_epoch']:
            lr = lr*float(1 + step + epoch*len_epoch)/(self.current_schedule['warmup_epoch']*len_epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, schedule=None):
        # 训练函数
        if schedule is None and self.global_schedule is None:
            # raise AttributeError("Training schedule is None, please check your schedule setting.")
            raise AttributeError("训练计划为None，请检查您的计划设置。")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        # 工作目录
        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        self.work_dir = work_dir

        # log and output:
        # 记录和输出:
        # 1. 实验配置
        # 2. 输出损失和时间
        # 3. 测试并输出统计信息
        # 4. 保存检查点

        log('==========计划参数==========\n')
        log(str(self.current_schedule)+'\n')

        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain'], map_location='cpu'), strict=False)
            log(f"Load pretrained parameters: {self.current_schedule['pretrain']}\n")

        # 使用GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========使用GPU进行训练==========\n')

            CUDA_VISIBLE_DEVICES = ''
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

            if CUDA_VISIBLE_DEVICES == '':
                # raise ValueError(f'This machine has no visible cuda devices!')
                raise ValueError('没有可用的CUDA设备！')

            CUDA_SELECTED_DEVICES = ''
            if 'CUDA_SELECTED_DEVICES' in self.current_schedule:
                CUDA_SELECTED_DEVICES = self.current_schedule['CUDA_SELECTED_DEVICES']
            else:
                CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

            CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
            CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
            if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
                # raise ValueError(f'CUDA_VISIBLE_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')
                raise ValueError('CUDA_SELECTED_DEVICES_SET 应该是 CUDA_VISIBLE_DEVICES 的子集！')

            GPU_num = len(CUDA_SELECTED_DEVICES_SET)
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)

            if GPU_num > 1:
                self.model = nn.DataParallel(self.model, device_ids=device_ids, output_device=device_ids[0])
        # 使用CPU
        else:
            device = torch.device("cpu")

        current_modelState = "良性模型"
        
        if self.current_schedule['benign_training'] is True:
            # 当启用良性训练时执行的操作
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
            current_modelState = "良性模型"
        elif self.current_schedule['benign_training'] is False:
            # 当启用恶意训练时执行的操作
            train_loader = DataLoader(
                self.poisoned_train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
            current_modelState = "毒化模型"
        else:
            # raise AttributeError("self.current_schedule['benign_training'] should be True or False.")
            raise AttributeError("benign_training 应该设置为 True 或者 False.")

        self.model.train()  # 设置为训练模式

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        iteration = 0
        last_time = time.time()

        # msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        msg = f"总训练样本数: {len(self.train_dataset)}\n总测试样本数: {len(self.test_dataset)}\n批处理大小: {self.current_schedule['batch_size']}\n每个周期的迭代次数: {len(self.train_dataset) // self.current_schedule['batch_size']}\n初始学习率: {self.current_schedule['lr']}\n"
        log(msg)

        for i in range(self.current_schedule['epochs']):
            for batch_id, batch in enumerate(train_loader):
                self.adjust_learning_rate(optimizer, i, batch_id, int(math.ceil(len(self.train_dataset) / self.current_schedule['batch_size'])))
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                # print(batch_img)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = self.model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                loss.backward()
                optimizer.step()

                iteration += 1

                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    # msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, 迭代次数:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, 学习率: {optimizer.param_groups[0]['lr']}, 损失: {float(loss)}, 耗时: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # test result on benign test dataset
                # 模型在良性数据集下测试的结果
                predict_digits, labels, mean_loss = self._test(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                # msg = "==========Test result on benign test dataset==========\n" + \
                #       time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                #       f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                msg = f"=========={current_modelState}在良性测试数据集上的测试结果==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Top-1 正确 / 总数: {top1_correct}/{total_num}, Top-1 准确率: {top1_correct/total_num}, Top-5 正确 / 总数: {top5_correct}/{total_num}, Top-5 准确率: {top5_correct/total_num}, 平均损失: {mean_loss}, 耗时: {time.time()-last_time}\n"

                log(msg)

                # test result on poisoned test dataset
                # 模型在毒化数据集下测试的结果
                # if self.current_schedule['benign_training'] is False:
                predict_digits, labels, mean_loss = self._test(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                # msg = "==========Test result on poisoned test dataset==========\n" + \
                #       time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                #       f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                msg = f"=========={current_modelState}在毒化测试数据集上的测试结果==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 正确 / 总数: {top1_correct}/{total_num}, Top-1 准确率: {top1_correct/total_num}, Top-5 正确 / 总数: {top5_correct}/{total_num}, Top-5 准确率: {top5_correct/total_num}, 平均损失: {mean_loss}, 耗时: {time.time()-last_time}\n"
                log(msg)

                self.model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                self.model.eval()
                torch.save(self.model.state_dict(), ckpt_model_path)  # 保存模型的字典
                # torch.save(self.model, ckpt_model_path)  # 保存整个模型
                self.model.train()

        self.model.eval()
        self.model = self.model.cpu()

    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None, test_loss=None):
        # 测试函数实现核心代码
        if model is None:
            model = self.model
        else:
            model = model

        if test_loss is None:
            test_loss = self.loss
        else:
            test_loss = test_loss

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model.eval()

            predict_digits = []
            labels = []
            losses = []
            for batch in test_loader:  # 遍历测试数据加载器中的每个批次
                batch_img, batch_label = batch  # 从当前批次中获取图像数据和对应的标签
                batch_img = batch_img.to(device)  # 将图像数据和标签移动到指定的计算设备（通常是 GPU）上，以便利用硬件加速进行模型推理
                batch_label = batch_label.to(device)  # 将图像数据和标签移动到指定的计算设备（通常是 GPU）上，以便利用硬件加速进行模型推理
                batch_img = model(batch_img)  # 通过模型进行图像数据的前向传播，得到模型的预测结果
                loss = test_loss(batch_img, batch_label)  # 计算模型在当前批次上的损失值

                predict_digits.append(batch_img.cpu()) # (B, self.num_classes)  将每个批次的预测结果、真实标签和损失值存储在相应的列表中。这里使用 cpu() 方法将张量移动回 CPU，以便在后续的计算中处理
                labels.append(batch_label.cpu()) # (B) 将每个批次的预测结果、真实标签和损失值存储在相应的列表中。这里使用 cpu() 方法将张量移动回 CPU，以便在后续的计算中处理
                if loss.ndim == 0: # scalar
                    loss = torch.tensor([loss])
                losses.append(loss.cpu()) # (B) or (1)

            predict_digits = torch.cat(predict_digits, dim=0) # (N, self.num_classes)
            labels = torch.cat(labels, dim=0) # (N)
            losses = torch.cat(losses, dim=0) # (N)
            return predict_digits, labels, losses.mean().item()

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None, test_loss=None):
        # 测试函数
        if schedule is None and self.global_schedule is None:
            # raise AttributeError("Test schedule is None, please check your schedule setting.")
            raise AttributeError("测试计划为空，请检查您的计划设置。")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        self.work_dir = work_dir

        # 使用GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========使用GPU进行测试==========\n')

            CUDA_VISIBLE_DEVICES = ''
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

            if CUDA_VISIBLE_DEVICES == '':
                raise ValueError('没有可用的CUDA设备！')

            CUDA_SELECTED_DEVICES = ''
            if 'CUDA_SELECTED_DEVICES' in self.current_schedule:
                CUDA_SELECTED_DEVICES = self.current_schedule['CUDA_SELECTED_DEVICES']
            else:
                CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

            CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
            CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
            if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
                raise ValueError(f'CUDA_ELECTED_DEVICES 应该是 CUDA_VISIBLE_DEVICES的子集！')

            GPU_num = len(CUDA_SELECTED_DEVICES_SET)
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)

            if GPU_num > 1:
                self.model = nn.DataParallel(self.model, device_ids=device_ids, output_device=device_ids[0])
        # 使用CPU
        else:
            device = torch.device("cpu")

        if self.current_schedule['benign_training'] == True:
            current_modelState = "良性模型"
        else:
            current_modelState = "毒性模型"
        
        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            # 在良性数据集上的测试结果
            predict_digits, labels, mean_loss = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model, test_loss)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            # msg = f"=========={current_modelState} 在良性数据集上的测试结果==========\n" + \
            #       time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            #       f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            msg = f"=========={current_modelState} 在良性数据集上的测试结果==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 正确 / 总数: {top1_correct}/{total_num}, Top-1 准确率: {top1_correct/total_num}, Top-5 正确 / 总数: {top5_correct}/{total_num}, Top-5 准确率: {top5_correct/total_num}, 平均损失: {mean_loss}, 耗时: {time.time()-last_time}\n"
            log(msg)
            return top1_correct, top5_correct, total_num, mean_loss

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            # 在毒化数据集上的测试结果
            predict_digits, labels, mean_loss = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model, test_loss)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            # msg = f"=========={current_modelState} 在毒化数据集上的测试结果==========\n" + \
            #       time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            #       f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            msg = f"=========={current_modelState} 在毒化数据集上的测试结果==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 正确 / 总数: {top1_correct}/{total_num}, Top-1 准确率: {top1_correct/total_num}, Top-5 正确 / 总数: {top5_correct}/{total_num}, Top-5 准确率: {top5_correct/total_num}, 平均损失: {mean_loss}, 耗时: {time.time()-last_time}\n"

            log(msg)
            return top1_correct, top5_correct, total_num, mean_loss
