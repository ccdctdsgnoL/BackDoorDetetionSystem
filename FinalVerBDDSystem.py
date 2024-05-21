"""
后门检测系统 2024.04.27
"""
import torch
from utils import *

@timer
def detect(select_model, model_path, master=None):
    ## 选择待检测模型
    model, dataset = selectModel(select_model, model_path)
    image_shape = dataset[0][0].shape  # 获取图像形状
    classes = dataset.classes

    # 触发器准备开始
    height, width = image_shape[1], image_shape[2]  # 获取图像高度和宽度
    ext_val = 8  # 触发器最大扩展值
    triggers = backDoorStyle(height, width, ext_val)
    print("初始触发器集形状：", triggers.shape)
    # 触发器准备完毕

    maxEpoch = 10 # 最大迭代次数
    for epoch in range(maxEpoch):
        print(f"\nEpoch: {epoch + 1}/{maxEpoch}")
        # 随机选取指定个数图像
        n = len(dataset) // len(triggers)
        batch_x, _ = randomChoiceData(dataset, len(triggers)*n)
        print("随机抽取数据形状：", batch_x.shape)
        poison_x = addTrigger(batch_x, triggers)

        normal_prediction_results = model.fc_l(batch_x)
        posion_prediction_results = model.fc_l(poison_x)

        diff_indices = torch.nonzero(normal_prediction_results != posion_prediction_results).squeeze()  # 获取分类改变的索引  核心代码
        classif_chage_prob = len(diff_indices) / len(batch_x) * 100  # 计算分类改变的概率
        colorPrint(f"[*] 分类改变个数：{len(diff_indices)}/{len(batch_x)}  分类改变概率：{classif_chage_prob:.2f}%", "blue")
        diff_tariggers = torch.index_select(posion_prediction_results, 0, diff_indices)  # 成功触发后被改变的标签
        triggers = triggers.repeat(n, 1, 1, 1)
        triggers = torch.index_select(triggers, 0, diff_indices)  # 影响到预测的后门样式
        
        # 统计每个数字出现的次数
        num_counts = torch.bincount(diff_tariggers)
        tariggers_len = len(diff_tariggers)
        print("每个数字出现的次数:", num_counts)
        
        flag = False
        # 获取从大到小排序的索引
        sorted_indices = torch.argsort(num_counts, descending=True)
        info_str = ""
        for i in sorted_indices:
            label = i.item()
            occ_prb = num_counts[i].item() / tariggers_len*100
            info_str += f"{label}: {occ_prb:.2f}%  "
            if occ_prb > 80 and classif_chage_prob > 60:
                flag = True
        colorPrint(f"[*] {info_str}", "blue")
        
        if flag:
            processed_tensor = process_triggers(triggers, 0.01)  # 堆叠后的后门图像
            colorPrint(f"[+] 检测到后门标签：{sorted_indices[0].item()}", "red")
            # show_tensor_image(processed_tensor, f"后门标签：{classes[sorted_indices[0].item()]}", "后门检测系统")
            start_GUI(processed_tensor, f"后门标签：{classes[sorted_indices[0].item()]}", "后门检测系统", model, dataset, master=master)

    if not flag:
        colorPrint("[-] 未检测出后门", "green") 
 
if __name__ == "__main__":
    detect("mnist", "E:\Code\My_python_files\BackDoorDetetionSystem\experiments\MNIST_2024-05-07_15-42-14\ckpt_epoch_10.pth")
