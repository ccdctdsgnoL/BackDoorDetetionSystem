import torch
import torch.nn.functional as F


class TestedModel:
    def __init__(self, model, model_path, batch_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.batch_size = batch_size
        chechkpoint = torch.load(model_path)
        self.model.load_state_dict(chechkpoint)
        self.model.to(self.device)
        self.model.eval()

    def fc_original(self, images):
        """
        返回原始预测结果
        """
        images = images.to(self.device)
        batch_size = self.batch_size  # 每批处理的图片数量
        # 分成多个批次处理
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        all_labels = []  # 存储所有批次的预测结果
        with torch.no_grad():
            for i in range(num_batches):
                # 获取当前批次的图片
                batch_images = images[i * batch_size : (i + 1) * batch_size]
                # 执行模型的前向传播    
                labels = self.model(batch_images)
                # 将预测结果添加到列表中
                all_labels.append(labels.cpu())
            # 合并所有批次的预测结果
            all_labels = torch.cat(all_labels, dim=0)
        
        return all_labels

    def fc_p(self, images):
        """
        返回预测概率
        """
        probabilities = F.softmax(self.fc_original(images), dim=1).cpu()  # 应用softmax函数转换到0-1之间
        return probabilities
    
    def fc_l(self, images):
        """
        返回预测标签
        """
        label = torch.argmax(self.fc_original(images), dim=1).cpu()
        return label