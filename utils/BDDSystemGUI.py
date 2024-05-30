import torch
import tkinter as tk
import numpy as np
from utils.BackDoorStyleTools import BackdoorStyleGenerator
from utils.showTensor import ImageViewer
from utils import randomChoiceData, addTrigger

class FvBDDSystemGUI(ImageViewer):
    def __init__(self, master, tensor, info=None, model=None, dataset=None):
        super().__init__(master, tensor, info)
        self.model = model
        self.dataset = dataset
        self.master = master
        
    def continue_run(self):
        subwindow = tk.Toplevel(self.master)
        self.bDSG = BackdoorStyleGUI(subwindow, self.tensor[self.current_index], self.model, self.dataset)
        self.bDSG.start()


class BackdoorStyleGUI(BackdoorStyleGenerator):
    def __init__(self, master, img_tensor=None, model=None, dataset=None):
        self.model = model
        self.dataset = dataset
        cell_size = 20
        img_tensor_dim = img_tensor.dim()
        grid_size = img_tensor.size(img_tensor_dim-1)
        
        self.root = master
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.canvas_size = grid_size * cell_size
        self.board = np.zeros((self.grid_size, self.grid_size))
        self.style_tensor = None
        self.root.title("Backdoor Style Generator")
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack()
        self.setup_grid()
        self.root.bind("<Escape>", lambda event: self.root.quit())  # 绑定Esc键退出程序
        self.canvas.bind("<Button-1>", self.on_click)  # 绑定鼠标左键点击事件
        self.canvas.bind("<B1-Motion>", self.on_drag)  # 绑定鼠标左键拖动事件
        self.canvas.bind("<Button-3>", lambda event: self.clear())  # 绑定鼠标右键点击事件
        self.root.bind("<Return>", lambda event: self.confirm_drawing())
        self.confirm_button = tk.Button(self.root, text=" 确认 ", command=self.confirm_drawing)  # 创建确认按钮
        self.confirm_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_button = tk.Button(self.root, text="清空画板", command=self.clear)
        self.clear_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.clear_enabled = False
        self.toggle_clear_button = tk.Button(self.root, text="启用小格清除", command=self.toggle_clear)
        self.toggle_clear_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.style_tensor = img_tensor
        self.display_tensor(img_tensor)

    def confirm_drawing(self):
        super().confirm_drawing()
        if self.model is not None:
            batch_x, real_label = randomChoiceData(self.dataset, 1000)
            batch_x_p = addTrigger(batch_x, self.style_tensor)
            labels = self.model.fc_l(batch_x_p)
            print("真实标签统计:", torch.bincount(real_label))
            pclass_counts = torch.bincount(labels)
            print("当前分类统计:", pclass_counts)
        else:
            print("未传入模型，无法统计分类统计")
        # print(labels)
        
        
def start_GUI(tensor, info=None, title=None, model=None, dataset=None, save_path=None, master=None):
    """ 
    tensor: [C, H, W] or [B, C, H, W] or [H, W]
    info: 展示信息 与图片个数相同，用于展示标签
        可接受类型：str list tensor
    title: 标题
    save_path: 保存路径
    """
    if master is None:
        root = tk.Tk()
        root.title(title) if title else root.title("Tensor Image")
        image_viewer = FvBDDSystemGUI(root, tensor, info, model, dataset)
        image_viewer.save_path = save_path
        root.mainloop()
    else:
        master = tk.Toplevel(master)
        image_viewer = FvBDDSystemGUI(master, tensor, info, model, dataset)
        image_viewer.save_path = save_path


if __name__ == '__main__':
    # test = FvBDDSystemGUI(None, None)
    # tensor = torch.randn(1, 1, 28, 28)
    # tensor = torch.zeros(28, 28)
    # tensor[0:4,0:4] = 1
    # app = BackdoorStyleGUI(tensor)
    # app.start()
    random_tensor = torch.randn(4, 1, 32, 32)
    start_GUI(random_tensor)
    