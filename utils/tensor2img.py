import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def save_tensor_image(tensor, filename):
    """
    保存 tensor 图片
    Args:
        tensor: 输入的 tensor 图片，应为三维张量 (C, H, W)
        filename: 要保存的文件名，包括路径和文件扩展名
    """
    # 如果输入张量是四维的，则取第一个样本
    if tensor.dim() == 4:
        tensor = tensor[0]
    # 将 tensor 转换为 PIL 图片
    image = torchvision.transforms.ToPILImage()(tensor)
    # 保存 PIL 图片
    image.save(filename)
    # print("Tensor image saved as", filename)

def show_tensor_image(tensor, num_images=16):
    """
    查看 tensor 图片
    Args:
        tensor: 输入的 tensor 图片，可以是三维张量 (C, H, W)，也可以是四维张量 (N, C, H, W)
    """
    if tensor.dim() == 3:
        # 如果输入是三维张量 (C, H, W)，则直接显示
        image = tensor.detach().permute(1, 2, 0).numpy()
        # plt.imshow(image)
        plt.imshow(image, cmap='gray')  # 设置颜色映射为灰度
        plt.axis('off')
        plt.show()
    elif tensor.dim() == 4:
        start_index = 0  # 起始索引
        while start_index < len(tensor):
            show_multi_images(tensor, start_index=start_index, num_images=num_images)
            response = input("Press Enter to show next page, or input 'q' to quit: ")
            if response.lower() == 'q':
                break
            start_index += num_images

def show_multi_images(tensor, start_index=0, num_images=16):
    tensor = tensor.detach().cpu().numpy()  # 将张量移动到CPU并转换为NumPy数组
    num_images = min(tensor.shape[0] - start_index, num_images)  # 选择要显示的图像数量
    rows = int(np.sqrt(num_images))  # 动态计算行数和列数
    cols = (num_images // rows) + int(num_images % rows > 0)
    tensor = tensor[start_index:start_index + num_images]  # 选择要显示的图像
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))  # 创建子图
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_images:
                # axes[i, j].imshow(tensor[i * cols + j][0])  # 显示图像
                axes[i, j].imshow(tensor[i * cols + j][0], cmap='gray')  # 显示图像
                axes[i, j].axis('off')  # 关闭坐标轴
    plt.show()


if __name__ == '__main__':
    # 示例用法
    tensor = torch.randn(3, 256, 256)  # 示例的 tensor 图片，假设为 256x256 大小，三通道 RGB 格式
    save_tensor_image(tensor, "tensor_image.jpg")  # 保存 tensor 图片
    show_tensor_image(tensor)  # 查看 tensor 图片
