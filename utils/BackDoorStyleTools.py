import torch
import numpy as np
import tkinter as tk

def backDoorStyle(height, width, ext_val=8):
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

def addTrigger(x, Delta, weight=1):
    """
    x: 原始图像
    Delta: 触发器
    weight: 触发器权重
    return: 添加触发器后的图像
    注意：x 和 Delta 必须是相同的大小
    """
    x_len = len(x)
    Delta_len = len(Delta)
    if x_len % Delta_len == 0:
        repeat_times = x_len // Delta_len
        Delta = Delta.repeat(repeat_times, 1, 1, 1)
    
    return Delta*weight + x*(1-Delta)

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
    # batch_indices = np.random.choice(dataset_len, n)
    # batch_x = torch.stack([dataset[i][0] for i in batch_indices])
    # batch_label = torch.stack([dataset[i][1] for i in batch_indices])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
    # 获取随机选取的n条数据
    for images, labels in data_loader:break
    # return batch_x
    return images, labels

def process_triggers(triggers, res_acc=0.1):
    """
    对触发器张量进行处理，并返回处理后的张量
    
    参数：
    - triggers: 形状为四维的触发器张量
    - res_acc: 分辨精度(每张图阈值相隔差距)
    
    返回：
    - processed_images_tensor: 经过处理后的触发器张量
    """
    # 对触发器张量进行求和
    summed_tensor = torch.sum(triggers, dim=0, keepdim=True)
    # 对触发器张量进行归一化处理
    max_value = torch.max(summed_tensor)
    process_img = (summed_tensor / max_value)
    iterator = np.arange(1.0-res_acc, 0.0, -res_acc)
    processed_images_tensor = torch.cat([torch.where(process_img < i, torch.tensor(0.0).unsqueeze(0), process_img) for i in iterator], dim=0)
    return processed_images_tensor



class BackdoorStyleGenerator:
    """
    后门样式生成器
    可以用来手动绘制后门样式，以生成特殊的图案
    鼠标右键快速清空
    回车键确定，按Esc退出
    """
    def __init__(self, grid_size=28, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.canvas_size = grid_size * cell_size
        self.board = np.zeros((self.grid_size, self.grid_size))
        self.style_tensor = None
        self.root = tk.Tk()
        self.root.title("Backdoor Style Generator")
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack()
        self.setup_grid()
        self.root.bind("<Escape>", lambda event: self.root.quit())  # 绑定Esc键退出程序
        self.canvas.bind("<Button-1>", self.on_click)  # 绑定鼠标左键点击事件
        self.canvas.bind("<B1-Motion>", self.on_drag)  # 绑定鼠标左键拖动事件
        self.canvas.bind("<Button-3>", lambda event: self.clear())  # 绑定鼠标右键点击事件
        # self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Return>", lambda event: self.confirm_drawing())
        self.confirm_button = tk.Button(self.root, text=" 确认 ", command=self.confirm_drawing)  # 创建确认按钮
        self.confirm_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_button = tk.Button(self.root, text="清空画板", command=self.clear)
        self.clear_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.clear_enabled = False
        self.toggle_clear_button = tk.Button(self.root, text="启用小格清除", command=self.toggle_clear)
        self.toggle_clear_button.pack(side=tk.RIGHT, padx=5, pady=5)
        # self.start()

    def setup_grid(self):  # 设置网格
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='black')

        # 画横线
        for i in range(1, self.grid_size):
            y = i * self.cell_size
            self.canvas.create_line(0, y, self.canvas_size, y, fill='green')

        # 画竖线
        for i in range(1, self.grid_size):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.canvas_size, fill='green')

    def on_click(self, event):  # 鼠标点击事件
        if self.clear_enabled:
            self.clear_cell(event)
        else:
            cell_x = event.x // self.cell_size
            cell_y = event.y // self.cell_size
            self.canvas.create_rectangle(cell_x * self.cell_size, cell_y * self.cell_size,
                                        (cell_x + 1) * self.cell_size, (cell_y + 1) * self.cell_size,
                                        fill='white', outline='white')
            self.board[cell_y, cell_x] = 1.0

    def on_drag(self, event):  # 鼠标拖动事件
        if self.clear_enabled:
            self.clear_cell(event)
        else:
            cell_x = event.x // self.cell_size
            cell_y = event.y // self.cell_size
            if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
                self.canvas.create_rectangle(cell_x * self.cell_size, cell_y * self.cell_size,
                                            (cell_x + 1) * self.cell_size, (cell_y + 1) * self.cell_size,
                                            fill='white', outline='white')
                self.board[cell_y, cell_x] = 1.0

        
    def clear_cell(self, event):  # 清除单元格
        if self.clear_enabled:
            cell_x = event.x // self.cell_size
            cell_y = event.y // self.cell_size
            if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
                self.canvas.create_rectangle(cell_x * self.cell_size, cell_y * self.cell_size,
                                            (cell_x + 1) * self.cell_size, (cell_y + 1) * self.cell_size,
                                            fill='black', outline='black')
                self.board[cell_y, cell_x] = 0.0

    def toggle_clear(self):  # 切换清除模式
        if self.clear_enabled:
            self.clear_enabled = False
            self.toggle_clear_button.config(text="启用小格清除")
        else:
            self.clear_enabled = True
            self.toggle_clear_button.config(text="禁用小格清除")

    def confirm_drawing(self):  # 确认绘制
        self.style_tensor = torch.tensor(self.board, dtype=torch.float32)
        # print(self.style_tensor)

    def clear(self):  # 清空画板
        self.board = np.zeros((self.grid_size, self.grid_size))
        self.canvas.delete("all")
        self.setup_grid()

    def get_tensor(self):  # 获取画板中的图像
        return self.style_tensor
    
    def set_tensor(self, tensor):
        self.style_tensor = tensor
    
    def start(self):
        self.root.mainloop()

    def display_tensor(self, tensor):  # 将张量显示到界面中的格子中
        while tensor.dim() > 2:
            tensor = tensor[0]
        self.board = tensor.clone().numpy()  # 更新self.board
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = tensor[i, j].item()
                fill_color = 'white' if value > 0.1 else 'black'
                self.canvas.create_rectangle(j * self.cell_size, i * self.cell_size,
                                            (j + 1) * self.cell_size, (i + 1) * self.cell_size,
                                            fill=fill_color, outline=fill_color)



if __name__ == "__main__":
    generator = BackdoorStyleGenerator(grid_size=28)  # 或者可以设置为32
    generator.set_tensor(torch.randn(32,32))
    tensor = generator.get_tensor()
    print(tensor)
