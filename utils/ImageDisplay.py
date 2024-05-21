
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class ImageDisplay:
    """
    使用matplotlib实现的图像查看
    是用来展示数据集里面的图像的
    能一次性展示16张图像和标签
    """

    # 图像显示
    def __init__(self, dataset):
        self.dataset = dataset
        # self.num_groups = num_groups
        self.total_images = len(dataset)
        self.num_iterations = self.total_images // 16
        self.current_group = 0
        self.fig, self.axes = plt.subplots(4, 4, figsize=(10, 10))
        self.fig.subplots_adjust(bottom=0.2)
        dataset_name = str(dataset.__class__).split('.')[-1][:-2]
        plt.gcf().canvas.set_window_title(dataset_name)  # 设置窗口标题
        self.prev_button = Button(plt.axes([0.4, 0.01, 0.1, 0.05]), '上一页')
        self.next_button = Button(plt.axes([0.55, 0.01, 0.1, 0.05]), '下一页')

        self.prev_button.on_clicked(self.prev_group)
        self.next_button.on_clicked(self.next_group)

    def show_images(self):
        start_index = self.current_group * 16
        for i in range(16):
            sample_index = start_index + i
            if sample_index >= self.total_images:
                break
            sample_image, sample_label = self.dataset[sample_index]
            # print(sample_image)

            row_index = i // 4
            col_index = i % 4

            self.axes[row_index, col_index].imshow(sample_image.squeeze().numpy(), cmap='gray')
            self.axes[row_index, col_index].set_title(f'Label: {sample_label}')
            self.axes[row_index, col_index].axis('off')

        plt.show()

    def prev_group(self, event):
        if self.current_group > 0:
            self.current_group -= 1
            self.show_images()

    def next_group(self, event):
        if self.current_group < self.num_iterations - 1:
            self.current_group += 1
            self.show_images()


if __name__ == '__main__':
    import torch
    random_tensor = torch.randn(128, 3, 32, 32)
    display_test = ImageDisplay(random_tensor)
    display_test.show_images()

    # 等待手动切换组
    # plt.show()
