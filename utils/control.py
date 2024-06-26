import torch
import threading
import time
import os
import ast
from utils import *
from PIL import Image, ImageTk
from torchvision.transforms import ToPILImage


class Controller:
    # 导入UI类后，替换以下的 object 类型，将获得 IDE 属性提示功能
    ui: object
    def __init__(self):
        self.to_pil = ToPILImage()
        self.Epochs = None  # 最大检测轮数
        self.IsEarlyStop = None  # 是否提前停止
        self.SelectModel = None  # 选择的模型
        self.ModelPath = None  # 模型路径
        self.SelectDataset = None  # 选择的数据集
        self.DatasetPath = None  # 数据集路径
        self.IsExistTrigger = None  # 是否存在后门
        self.CurrentDataSetImage = None  # 当前数据集图像
        self.CurrentDatasetIndex = 0  # 当前数据集索引
        self.CurrentTriggerImage = None  # 当前触发器图像
        self.CurrentTriggerIndex = 0  # 当前触发器索引
        self.currentBackDoorlabel = None  # 当前后门标签
        self.current_posion_prediction_results = None  # 当前毒化数据预测结果
        self.IsAccelerate = False  # 是否加速
        self.CustomModelFilePath = None  # 自定义模型文件路径
        self.CustomModelCode = None  # 自定义模型代码
        self.CustomModelLoadMethod = None  # 自定义模型加载方法
        self.IsUseCustomModel = False  # 是否使用自定义模型
        self.CustomModel = None  # 自定义模型
        self.CustomDatasetPath = None  # 自定义数据集路径
        self.CurrentUseModel = None  # 当前使用的模型
        self.CurrentUseDataset = None  # 当前使用的数据集
        self.IsUseCustomDataset = False  # 是否使用自定义数据集

    
    def init(self, ui):
        """
        得到UI实例，对组件进行初始化配置
        """
        self.ui = ui
        # TODO 组件初始化 赋值操作
    
    def getUIValue(self):
        self.Epochs = self.ui.Epochs.get()  # 最大检测轮数
        self.IsEarlyStop = self.ui.IsEarlyStop.get()  # 是否提前停止
        self.SelectModel = self.ui.tk_select_box_SelectNetModel.get()  # 选择的模型
        self.ModelPath = self.ui.tk_input_ModelPath.get()  # 模型路径
        self.SelectDataset = self.ui.tk_select_box_SelectDataset.get()  # 选择的数据集
        self.DatasetPath = self.ui.tk_input_DatasetPath.get()  # 数据集路径
        self.IntervalAcc = self.ui.IntervalAcc.get()  # 间隔精度
        if self.IsUseCustomDataset:
            self.CustomDatasetPath = self.ui.tk_input_DatasetPath.get()
        else:
            self.CustomDatasetPath = self.ui.tk_select_box_SelectDataset.get()
        
    def StartDetect(self, evt):
        self.ui.tk_label_DatasetImage.config(text="毒化数据")
        self.ui.tk_label_ShowDatasetImage.config(image="")
        self.ui.tk_label_BackDoorLabel.config(text="后门标签")
        self.ui.tk_label_ShowBackDoorImage.config(image="")
        self.ui.progress_value.set(0)
        self.ui.tk_label_Process.config(text="进度")
        self.CurrentTriggerImage = None
        self.CurrentTriggerIndex = 0
        self.currentBackDoorlabel = None
        self.current_normal_prediction_results = None
        self.current_posion_prediction_results = None
        self.CurrentDataSetImage = None
        self.CurrentDatasetIndex = 0
        self.CurrentUseModel = None  # 当前使用的模型
        self.CurrentUseDataset = None  # 当前使用的数据集
        self.getUIValue()
        # 创建一个新线程来执行耗时任务
        detect_thread = threading.Thread(target=self.run_detection)
        detect_thread.start()
        
    def update_epoch_label(self, event):
        self.ui.tk_label_SetEpoch.config(text=f"最大检测轮数：{int(self.ui.tk_scale_SetEpoch.get())}")

    def on_dataset_select(self, event):
        if self.ui.tk_select_box_SelectDataset.get() == "其他数据集":
            self.ui.tk_input_DatasetPath.place(relx=0.6750, rely=0.3333, relwidth=0.3225, relheight=0.2500)
            self.IsUseCustomDataset = True  # 是否使用自定义数据集
        else:
            self.IsUseCustomDataset = False  # 是否使用自定义数据集
            self.CustomDatasetPath = self.ui.tk_select_box_SelectDataset.get()  # 自定义数据集路径
            self.ui.tk_input_DatasetPath.place_forget()
    
    def on_net_model_select(self, event):
        if self.ui.tk_select_box_SelectNetModel.get() == "自定义模型":
            # self.ui.messagebox.showinfo("提示", "正在完善此功能")
            self.ui.subwin_CustomModel = self.ui.run_tk_toplevel_CustomModel(self.ui)
            # if self.CustomModelFilePath is not None:
            #     self.ui.subwin_CustomModel.tk_input_NetModelFilePath.set(self.CustomModelFilePath)
            if self.CustomModelCode is not None:
                self.ui.subwin_CustomModel.tk_text_NetModelCode.delete(1.0, "end")
                self.ui.subwin_CustomModel.tk_text_NetModelCode.insert(1.0, self.CustomModelCode)
        else:
            self.IsUseCustomModel = False
    
    def show_dataset_image(self):
        if self.CurrentDataSetImage is None:
            return
        self.show_image(self.ui.tk_label_DatasetImage, self.ui.tk_label_ShowDatasetImage,
                        self.CurrentDataSetImage, self.CurrentDatasetIndex, 
                        f"{self.dataset_classes[self.current_normal_prediction_results[self.CurrentDatasetIndex].item()]}->{self.dataset_classes[self.current_posion_prediction_results[self.CurrentDatasetIndex].item()]}"
                        )
        
    def on_dataset_prvious(self, event):
        self.CurrentDatasetIndex -= 1
        self.show_dataset_image()

    def on_dataset_next(self, event):
        self.CurrentDatasetIndex += 1
        self.show_dataset_image()
    
    def show_backdoor_image(self):
        if self.CurrentTriggerImage is None:
            return
        self.show_image(self.ui.tk_label_BackDoorLabel, self.ui.tk_label_ShowBackDoorImage,
                        self.CurrentTriggerImage, self.CurrentTriggerIndex, self.currentBackDoorlabel)
    
    def on_backdoor_img_prvious(self, event):
        self.CurrentTriggerIndex -= 1
        self.show_backdoor_image()
    
    def on_backdoor_img_next(self, event):
        self.CurrentTriggerIndex += 1
        self.show_backdoor_image()
    
    def on_verify(self, event):
        if self.CurrentTriggerImage is None:
            return
        start_GUI(self.CurrentTriggerImage, f"后门标签：{self.currentBackDoorlabel}", "后门检测系统", self.CurrentUseModel, self.CurrentUseDataset, master=self.ui)
        
    def show_image(self, info_label, image_label, tensor, current_index, info=None):
        pic_num = 1
        if tensor.dim() == 2:
            tensor.unsqueeze(0)
            image_array = tensor
        elif tensor.dim() == 3:
            image_array = tensor
        elif tensor.dim() == 4:
            pic_num = len(tensor)
            image_array = tensor[current_index%pic_num]
        if type(info) == str:
            info_label.config(text=info)
        image = self.to_pil(image_array)
        self.now_image = image
        min_pix = min(image_label.winfo_width(), image_label.winfo_height())
        new_width ,new_height = min_pix, min_pix
        image = image.resize((new_width, new_height), Image.NEAREST)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        
    def run_detection(self):
        start_time = time.time()
        if self.ModelPath is None or self.ModelPath == "":
            self.ui.messagebox.showerror("错误", "请填写模型路径")
            return
        if self.IntervalAcc < 0 or self.IntervalAcc > 1:
            self.ui.IntervalAcc.set(0.1)
            self.IntervalAcc = 0.1
            self.ui.messagebox.showinfo("提示", "间隔精度范围为0-1，已自动设置为0.1")
            return
        print(f"\n[*] 开始检测，最大检测轮数：{self.Epochs}，是否提前停止：{self.IsEarlyStop}")
        self.ui.tk_button_Start.config(text="正在检测...")
        try:
            ## 选择待检测模型
            model, dataset = selectModel(self.SelectModel, self.ModelPath, self.CustomModel, self.CustomDatasetPath)
            self.CurrentUseModel = model
            self.CurrentUseDataset = dataset
            print(self.CurrentUseDataset)
            image_shape = dataset[0][0].shape  # 获取图像形状
            classes = dataset.classes
            self.dataset_classes = classes

            # 触发器准备开始
            height, width = image_shape[1], image_shape[2]  # 获取图像高度和宽度
            ext_val = 8  # 触发器最大扩展值
            triggers = backDoorStyle(height, width, ext_val)
            print("初始触发器集形状：", triggers.shape)
            # 触发器准备完毕

            maxEpoch = self.Epochs  # 最大迭代次数
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
                output_info = f"[*] 分类改变个数：{len(diff_indices)}/{len(batch_x)}  分类改变概率：{classif_chage_prob:.2f}%"
                colorPrint(output_info, "blue")
                diff_tariggers = torch.index_select(posion_prediction_results, 0, diff_indices)  # 成功触发后被改变的标签
                triggers = triggers.repeat(n, 1, 1, 1)
                triggers = torch.index_select(triggers, 0, diff_indices)  # 影响到预测的后门样式
                
                self.current_normal_prediction_results = torch.index_select(normal_prediction_results, 0, diff_indices)
                self.current_posion_prediction_results = torch.index_select(posion_prediction_results, 0, diff_indices)
                self.CurrentDataSetImage = torch.index_select(poison_x, 0, diff_indices)
                self.show_dataset_image()

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
                self.ui.progress_value.set((epoch + 1)/maxEpoch*100)
                self.ui.tk_label_Process.config(text=f"{self.ui.progress_value.get():.2f}%")
                processed_tensor = process_triggers(triggers, self.IntervalAcc)  # 堆叠后的后门图像拆分出来
                
                if flag:
                    triggers = torch.cat((triggers, processed_tensor), dim=0)
                    triggers = triggers[torch.randperm(triggers.size(0))]  # 随机打乱后门样式
                    colorPrint(f"[+] 检测到后门标签：{sorted_indices[0].item()}", "red")
                    self.currentBackDoorlabel = classes[sorted_indices[0].item()]
                    self.CurrentTriggerImage = processed_tensor
                    self.show_backdoor_image()
                    if self.IsEarlyStop:
                        break
                elif self.IsAccelerate:
                    triggers = torch.cat((triggers, processed_tensor), dim=0)
                    triggers = triggers[torch.randperm(triggers.size(0))]  # 随机打乱后门样式

            if not flag:
                colorPrint("[-] 未检测出后门", "green")
                end_time = time.time()
                spend_time = end_time - start_time
                self.ui.messagebox.showinfo("未发现后门", f"耗时：{spend_time:.2f}s\n{output_info}\n可以查看控制台考虑增加最大检测轮数")
            else:
                end_time = time.time()
                spend_time = end_time - start_time
                self.ui.messagebox.showwarning("发现后门！", f"检测完成, 耗时：{spend_time:.2f}s\n发现后门标签：{self.currentBackDoorlabel}")
                # start_GUI(processed_tensor, f"后门标签：{classes[sorted_indices[0].item()]}", "后门检测系统", model, dataset, master=self.ui)
        except Exception as e:
            colorPrint(f"[-] 检测出异常：{e}", "red")
            self.ui.messagebox.showerror("错误", f"检测时发生了错误：{e}")
        finally:
            self.ui.tk_button_Start.config(text="开始检测")
    """
    以下是自定义模型窗口的事件处理代码
    self.ui.subwin_CustomModel
    """
    def LoadModelCodeFile(self, evt):
        self.CustomModelFilePath = self.ui.subwin_CustomModel.tk_input_NetModelFilePath.get()
        if os.path.exists(self.CustomModelFilePath):
            print("载入文件：", self.CustomModelFilePath)
            with open(self.CustomModelFilePath, "r", encoding="utf-8") as f:
                self.CustomModelCode = f.read()
            self.ui.subwin_CustomModel.tk_text_NetModelCode.delete(1.0, "end")
            self.ui.subwin_CustomModel.tk_text_NetModelCode.insert(1.0, self.CustomModelCode)
        else:
            print(self.CustomModelFilePath, " 文件不存在")
            self.ui.subwin_CustomModel.messagebox.showerror("错误", "文件不存在")
        
    def VerifyLoadMethod(self, evt):
        print("验证加载代码")
        print(self.ui.subwin_CustomModel.tk_input_LoadMethod.get())
        self.CustomModelLoadMethod = self.ui.subwin_CustomModel.tk_input_LoadMethod.get()
        
        # 执行模型代码
        exec(self.CustomModelCode, globals())
        # 尝试实例化模型
        try:
            model = eval(self.CustomModelLoadMethod)
            print("实例化模型：", model)
            self.ui.subwin_CustomModel.messagebox.showinfo("模型信息", model)
            self.CustomModel = model
        except Exception as e:
            print(f"实例化方法错误：{e}")
            self.ui.subwin_CustomModel.messagebox.showerror("错误", f"实例化方法错误：{e}")
            return
        
    def CancelLoadNetModel(self, evt):
        print("取消加载模型")
        self.CustomModelFilePath = None
        self.CustomModelCode = None
        self.CustomModelLoadMethod = None
        self.IsUseCustomModel = False
        self.CustomModel = None
        self.ui.subwin_CustomModel.destroy()
        
    def CertainLoadNetModel(self, evt):
        print("确定加载模型")
        self.IsUseCustomModel = True
        self.ui.subwin_CustomModel.destroy()