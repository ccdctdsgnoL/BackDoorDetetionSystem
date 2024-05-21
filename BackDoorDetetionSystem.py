"""
后门检测系统一键运行GUI
"""

import tkinter as tk
from tkinter import ttk
import threading
from FinalVerBDDSystem import detect as backdoor_detection

class NeuralNetworkBackdoorDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("神经网络后门检测系统")
        self.root.geometry("600x400")  # 设置窗口默认大小

        # 模式切换变量
        self.mode_var = tk.StringVar(value="后门检测模式")

        # 模型结构选择
        self.model_structure_var = tk.StringVar()
        self.model_structure_label = tk.Label(self.root, text="模型结构：")
        self.model_structure_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)
        self.model_structure_combobox = ttk.Combobox(self.root, textvariable=self.model_structure_var, values=["mnist", "fashion_mnist", "cifar10", "自定义"])
        self.model_structure_combobox.grid(row=0, column=1, padx=10, pady=10)
        self.model_structure_combobox.bind("<<ComboboxSelected>>", self.on_model_structure_selected)
        self.model_structure_text = tk.Text(self.root, height=10, width=50)

        # 模型路径输入
        self.model_path_var = tk.StringVar()
        self.model_path_label = tk.Label(self.root, text="模型路径：")
        self.model_path_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)
        self.model_path_entry = tk.Entry(self.root, textvariable=self.model_path_var)
        self.model_path_entry.grid(row=1, column=1, padx=10, pady=10)

        # 数据集选择
        self.dataset_option_var = tk.IntVar()
        self.dataset_option_var.set(1)  # 默认自动下载数据集
        self.auto_download_radio = tk.Radiobutton(self.root, text="自动加载数据集", variable=self.dataset_option_var, value=1, command=self.toggle_dataset_input)
        self.auto_download_radio.grid(row=2, column=0, sticky="w", padx=10, pady=10)
        self.manual_input_radio = tk.Radiobutton(self.root, text="手动输入数据集路径", variable=self.dataset_option_var, value=2, command=self.toggle_dataset_input)
        self.manual_input_radio.grid(row=3, column=0, sticky="w", padx=10, pady=10)
        self.dataset_path_var = tk.StringVar()
        self.dataset_path_entry = tk.Entry(self.root, textvariable=self.dataset_path_var, state="disabled")
        self.dataset_path_entry.grid(row=3, column=1, padx=10, pady=10)

        # 检测按钮
        self.detect_button = tk.Button(self.root, text="检测", command=self.detect)
        self.detect_button.grid(row=4, column=0, padx=10, pady=10)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        # 模式切换按钮
        self.mode_button = tk.Button(self.root, text="切换到后门模型生成模式", command=self.toggle_mode)
        self.mode_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)


    def toggle_dataset_input(self):
        if self.dataset_option_var.get() == 1:
            self.dataset_path_entry.config(state="disabled")
        else:
            self.dataset_path_entry.config(state="normal")

    def on_model_structure_selected(self, event):
        if self.model_structure_var.get() == "自定义":
            # 如果选择自定义，可以在这里添加自定义模型结构的输入框
            self.model_structure_text.grid(row=7, column=0, columnspan=2, padx=10, pady=50)
        else:
            # 如果不是自定义，可以移除或禁用自定义模型结构的输入框
            self.model_structure_text.grid_remove()
            pass

    def detect(self):
        print("模型结构：", self.model_structure_var.get())
        print("模型路径：", self.model_path_var.get())
        if self.dataset_option_var.get() == 1:
            print("数据集：自动加载")
        else:
            print("数据集路径：", self.dataset_path_var.get())
        threading.Thread(target=backdoor_detection, args=(self.model_structure_var.get(), self.model_path_var.get(), self.root)).start()

    def toggle_mode(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkBackdoorDetector(root)
    root.mainloop()
