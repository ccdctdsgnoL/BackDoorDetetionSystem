import torch
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk
from torchvision.transforms import ToPILImage
from time import sleep

class ImageViewer:
    """
    张量图片查看器
    能够将二、三、四维张量转换为图片，并显示
    """
    def __init__(self, master, tensor, info=None):
        self.master = master
        self.tensor = tensor
        self.save_path = None
        self.info = info
        self.is_playing = False
        self.current_index = 0
        self.zoom_factor = 10.0
        self.to_pil = ToPILImage()
        
        self.info_label = tk.Label(master, text="")
        self.info_label.pack()

        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.index_label = tk.Label(master, text="")
        self.index_label.pack()

        self.prev_button = tk.Button(master, text="上一张", command=self.show_previous_image)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(master, text="下一张", command=self.show_next_image)
        self.next_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(master, text="保存", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)

        self.zoom_in_button = tk.Button(master, text="放大", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT)

        self.zoom_out_button = tk.Button(master, text="缩小", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT)
        
        self.continue_button = tk.Button(master, text="继续", command=self.continue_run)
        self.continue_button.pack(side=tk.LEFT)
        
        self.quit_button = tk.Button(master, text="结束", command=self.end_run)
        self.quit_button.pack(side=tk.LEFT)

        self.show_image()

    def show_image(self):  # 显示图像
        pic_num = 1
        if self.tensor.dim() == 2:
            self.tensor.unsqueeze(0)
            image_array = self.tensor
        elif self.tensor.dim() == 3:
            image_array = self.tensor
        elif self.tensor.dim() == 4:
            image_array = self.tensor[self.current_index]
            pic_num = len(self.tensor)
        if type(self.info) == str:
            self.info_label.config(text=self.info)
        elif type(self.info) == list or isinstance(self.info, torch.Tensor):
            if len(self.info) == pic_num:
                self.info_label.config(text=self.info[self.current_index])
            else:
                self.info_label.config(text="传入数据错误")
        image = self.to_pil(image_array)
        self.now_image = image
        new_width = int(image.width * self.zoom_factor)
        new_height = int(image.height * self.zoom_factor)
        image = image.resize((new_width, new_height), Image.NEAREST)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.index_label.config(text=f"第{self.current_index+1}张/总共{pic_num}张")

    def show_previous_image(self):  # 查看上一张图片
        self.current_index = (self.current_index - 1) % len(self.tensor)
        self.show_image()

    def show_next_image(self):  # 查看下一张图片
        self.current_index = (self.current_index + 1) % len(self.tensor)
        self.show_image()

    def save_image(self):  # 保存图片
        if self.save_path:
            self.now_image.save(f"{self.save_path}\\image_{self.current_index}_{self.now_time()}.png")
        else:
            self.now_image.save(f"image_{self.current_index}_{self.now_time()}.png")

    def zoom_in(self):  # 放大
        self.zoom_factor *= 1.1
        self.show_image()

    def zoom_out(self):  # 缩小
        self.zoom_factor *= 0.9
        self.show_image()
    
    def continue_run(self):  # 继续
        self.master.quit()
        
    def end_run(self):  # 结束运行
        exit()
    
    def play(self):
        if not self.is_playing:
            print("开始播放")
            for i in range(len(self.tensor)):
                print("第", i, "张")
                self.show_next_image()
                sleep(1)
    
    def now_time(self):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        return formatted_time

def show_tensor_image(tensor, info=None, title=None, save_path=None, master=None):
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
        image_viewer = ImageViewer(root, tensor, info)
        image_viewer.save_path = save_path
        root.mainloop()
    else:
        image_viewer = ImageViewer(master, tensor, info)
        image_viewer.save_path = save_path


if __name__ == '__main__':
    random_tensor = torch.randn(4, 1, 32, 32)
    trigger = torch.zeros(1, 32, 32)
    trigger[0,-4:,-4:] = 1
    show_tensor_image(trigger)
