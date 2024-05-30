import torch
from utils import *
import threading
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
        
    def StartDetect(self, evt):
        self.ui.tk_label_DatasetImage.config(text="毒化数据")
        self.ui.tk_label_ShowDatasetImage.config(image="")
        self.ui.tk_label_BackDoorLabel.config(text="后门标签")
        self.ui.tk_label_ShowBackDoorImage.config(image="")
        self.ui.progress_value.set(0)
        self.ui.tk_label_Process.config(text="进度")
        # 创建一个新线程来执行耗时任务
        detect_thread = threading.Thread(target=self.run_detection)
        detect_thread.start()
        
    def update_epoch_label(self, event):
        self.ui.tk_label_SetEpoch.config(text=f"最大检测轮数：{int(self.ui.tk_scale_SetEpoch.get())}")

    def on_dataset_select(self, event):
        if self.ui.tk_select_box_SelectDataset.get() == "其他数据集":
            self.ui.tk_input_DatasetPath.place(relx=0.6750, rely=0.3333, relwidth=0.3225, relheight=0.2500)
        else:
            self.ui.tk_input_DatasetPath.place_forget()
    
    def on_net_model_select(self, event):
        if self.ui.tk_select_box_SelectNetModel.get() == "自定义模型":
            self.ui.messagebox.showinfo("提示", "暂无此功能")
        else:
            pass
    
    def show_dataset_image(self):
        self.show_image(self.ui.tk_label_DatasetImage, self.ui.tk_label_ShowDatasetImage,
                        self.CurrentDataSetImage, self.CurrentDatasetIndex, self.current_posion_prediction_results)
        
    def on_dataset_prvious(self, event):
        self.CurrentDatasetIndex -= 1
        self.show_dataset_image()

    def on_dataset_next(self, event):
        self.CurrentDatasetIndex += 1
        self.show_dataset_image()
    
    def show_backdoor_image(self):
        self.show_image(self.ui.tk_label_BackDoorLabel, self.ui.tk_label_ShowBackDoorImage,
                        self.CurrentTriggerImage, self.CurrentTriggerIndex, self.currentBackDoorlabel)
    
    def on_backdoor_img_prvious(self, event):
        self.CurrentTriggerIndex -= 1
        self.show_backdoor_image()
    
    def on_backdoor_img_next(self, event):
        self.CurrentTriggerIndex += 1
        self.show_backdoor_image()
        
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
        elif type(info) == list:
            if len(info) == pic_num:
                info_label.config(text=self.dataset_classes[info[current_index]])
            else:
                info_label.config(text="传入数据错误")
        elif isinstance(info, torch.Tensor):
            if info.dim() > 0:
                if len(info) == pic_num:
                    info_label.config(text=self.dataset_classes[info[current_index].item()])
                else:
                    info_label.config(text="传入数据错误")
            else:
                info_label.config(text=info.item())
        image = self.to_pil(image_array)
        self.now_image = image
        min_pix = min(image_label.winfo_width(), image_label.winfo_height())
        new_width ,new_height = min_pix, min_pix
        image = image.resize((new_width, new_height), Image.NEAREST)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        
    def run_detection(self):
        self.getUIValue()  # 获取UI组件的值
        print(f"\n[*] 开始检测，最大检测轮数：{self.Epochs}，是否提前停止：{self.IsEarlyStop}")
        self.ui.tk_button_Start.config(text="正在检测...")
        try:
            ## 选择待检测模型
            model, dataset = selectModel(self.SelectModel, self.ModelPath)
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
                colorPrint(f"[*] 分类改变个数：{len(diff_indices)}/{len(batch_x)}  分类改变概率：{classif_chage_prob:.2f}%", "blue")
                diff_tariggers = torch.index_select(posion_prediction_results, 0, diff_indices)  # 成功触发后被改变的标签
                triggers = triggers.repeat(n, 1, 1, 1)
                triggers = torch.index_select(triggers, 0, diff_indices)  # 影响到预测的后门样式
                
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
                if flag:
                    processed_tensor = process_triggers(triggers, 0.01)  # 堆叠后的后门图像拆分出来
                    colorPrint(f"[+] 检测到后门标签：{sorted_indices[0].item()}", "red")
                    self.currentBackDoorlabel = classes[sorted_indices[0].item()]
                    self.CurrentTriggerImage = processed_tensor
                    self.show_backdoor_image()
                    if self.IsEarlyStop:
                        break

            if not flag:
                colorPrint("[-] 未检测出后门", "green")
            else:
                self.ui.messagebox.showinfo("提示", "检测完成")
                start_GUI(processed_tensor, f"后门标签：{classes[sorted_indices[0].item()]}", "后门检测系统", model, dataset, master=self.ui)
            
        except Exception as e:
            colorPrint(f"[-] 检测出异常：{e}", "red")
            self.ui.messagebox.showerror("错误", f"检测时发生了错误：{e}")
        finally:
            self.ui.tk_button_Start.config(text="开始检测")
