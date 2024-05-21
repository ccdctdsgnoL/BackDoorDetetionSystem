from tkinter import Tk, Label, Canvas, BOTH, W, N, E, S, Toplevel
from PIL import Image, ImageTk
import os

def load_image(image_path):
    return Image.open(image_path)

def resize_image(image, scale):
    width, height = image.size
    return image.resize((width * scale, height * scale))

def concatenate_images(images, direction='horizontal'):
    widths, heights = zip(*(i.size for i in images))
    if direction == 'horizontal':
        total_width = sum(widths)
        max_height = max(heights)
    else:
        total_width = max(widths)
        total_height = sum(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        if direction == 'horizontal':
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        else:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
    return new_im

def on_drag_files(files):
    images = [load_image(file) for file in files]
    resized_images = [resize_image(im, 10) for im in images]
    concatenated_image = concatenate_images(resized_images)
    imageTk = ImageTk.PhotoImage(concatenated_image)
    label = Label(canvas, image=imageTk)
    label.image = imageTk
    label.pack(fill=BOTH, expand=True)

root = Tk()
canvas = Canvas(root, width=800, height=200)
canvas.pack(fill=BOTH, expand=True)
label = Label(canvas, text="拖放MNIST图片到这里")
label.pack(fill=BOTH, expand=True)

# 设置拖放文件回调
root.bind('<B1-Motion>', lambda event: None)
root.bind('<Button-1>', lambda event: None)
root.bind('<B1-ButtonRelease>', lambda event: None)
root.bind('<Drop>', lambda event: on_drag_files(event.data))

root.mainloop()
