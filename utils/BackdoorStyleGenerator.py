"""
后门样式生成器
可以用来手动绘制后门样式，以生成特殊的图案
鼠标右键快速清空
回车键确定，按Esc退出
"""

import numpy as np
import tkinter as tk
import torch


class BackdoorStyleGenerator:
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
        self.root.mainloop()

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
        print(self.style_tensor)

    def clear(self):  # 清空画板
        self.board = np.zeros((self.grid_size, self.grid_size))
        self.canvas.delete("all")
        self.setup_grid()

    def get_tensor(self):  # 获取画板中的图像
        return self.style_tensor


if __name__ == "__main__":
    generator = BackdoorStyleGenerator(grid_size=28)  # 或者可以设置为32
    tensor = generator.get_tensor()
    print(tensor)
