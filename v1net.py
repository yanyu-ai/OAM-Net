import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import cv2
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, f1_score, recall_score, \
    roc_auc_score
import matplotlib.pyplot as plt
from dataset_paths import TRAIN_PATH, TEST_PATH
import random
import math

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

IMG_CLS = ["l=1", "l=2"]
IMG_SIZE = 200  # 统一大小的尺寸

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, paths, img_size):
        self.images, self.labels = self.load_data(paths, img_size)

    def load_data(self, paths, img_size):
        data_images = []
        data_labels = []
        for i, dir in enumerate(paths):  # 外层循环：enumerate(paths) 返回路径列表中的索引和路径，i 是索引，dir 是路径
            path_dir = os.listdir(dir)  # 列出路径 dir 下的所有文件和子目录
            for file in path_dir:  # 内层循环：遍历 path_dir 中的每个文件
                img_dir = os.path.join(dir, file)
                img = cv2.resize(cv2.imread(img_dir), (img_size, img_size)) / 255.0  # 读取图像。缩放到img_size尺寸并归一化
                data_images.append(img)  # 将处理后的图像添加到 data_images 列表
                labels = np.zeros(len(paths))
                labels[i] = 1.0  # 将当前路径对应的索引位置的标签设为1，表示图像属于该类别。
                data_labels.append(labels)  # 将标签添加到 data_labels 列表。
        data_images = np.array(data_images).astype("float32")
        data_labels = np.array(data_labels).astype("float32")
        return data_images, data_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image).permute(2, 0, 1)  # HWC -> CHW
        label = torch.tensor(label)
        return image, label


# 创建数据集和数据加载器
dataset = CustomDataset(TRAIN_PATH, IMG_SIZE)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = CustomDataset(TEST_PATH, IMG_SIZE)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义ECA模块
class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out

# 定义深度可分离卷积
def conv_dw(in_channel, out_channel, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel, bias=False),
        nn.BatchNorm2d(in_channel),
        ECA_block(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        ECA_block(out_channel),
        nn.ReLU(inplace=True)
    )


# 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = conv_dw(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv_dw(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv_dw(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(12*12*256, 512)  # 调整全连接层的输入大小
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 15)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(-1, 12*12*256)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 超参数
LEARNING_RATE = 0.0002  # 学习率
TRAIN_EPOCH = 40  # 训练轮数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型、损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_start_time = time.time()  # 记录训练开始时间

# 创建一个文件来记录损失和准确率
log_file = open("V1Net-NO-ECA.txt", "w")
log_file.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")

# 训练模型
for epoch in range(TRAIN_EPOCH):
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)  # 移动到GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        total_train_correct += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / len(train_dataset)

    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            total_val_loss += loss.item()
            total_val_correct += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_val_correct / len(val_dataset)

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 将每个epoch的损失和准确率写入文件
    log_file.write(f"{epoch + 1},{avg_train_loss:.4f},{train_accuracy:.4f},{avg_val_loss:.4f},{val_accuracy:.4f}\n")

log_file.close()  # 关闭文件

train_end_time = time.time()  # 记录训练结束时间
print(f"Total Training Time: {train_end_time - train_start_time:.2f}s")

# 测试模型
model.eval()
total_correct = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)  # 移动到GPU
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        true_labels = torch.argmax(labels, 1)
        total_correct += (preds == true_labels).sum().item()
        all_labels.extend(true_labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

accuracy = total_correct / len(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")

# 计算多分类的精确率、召回率、F1分数
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
print(f'Precision (macro): {precision:.4f}')
print(f'Recall (macro): {recall:.4f}')
print(f'F1 Score (macro): {f1:.4f}')

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

# 计算模型占用的内存（以MB为单位）
param_size_in_bytes = total_params * 4  # 每个参数占用4字节（float32）
param_size_in_mb = param_size_in_bytes / (1024 ** 2)  # 转换为MB
print(f"Model Memory Usage: {param_size_in_mb:.2f} MB")


# 对于 AUC 计算，需要将标签进行二值化
# n_classes = 15
# all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14])
# all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14])
#
# # 计算 AUC (ROC-AUC值)
# auc = roc_auc_score(all_labels_bin, all_preds_bin, average="macro", multi_class="ovr")
# print(f'AUC: {auc:.4f}')


# # 计算混淆矩阵，40分类
# cm = confusion_matrix(all_labels, all_preds)
#
# # 所有类别标签（1~40）
# labels = [str(i) for i in range(1, 41)]
#
# # 提取5的倍数的坐标索引和标签
# tick_indices = [i for i in range(40) if (i + 1) % 5 == 0]
# tick_labels = [labels[i] for i in tick_indices]
#
# # 设置加粗的罗马字体
# font_ticks = {'family': 'serif', 'name': 'Times New Roman', 'size': 16, 'weight': 'bold'}
# font_title = {'family': 'serif', 'name': 'Times New Roman', 'size': 18, 'weight': 'bold'}
# font_axis = {'family': 'serif', 'name': 'Times New Roman', 'size': 17, 'weight': 'bold'}
#
# # 绘制混淆矩阵
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# fig, ax = plt.subplots(figsize=(10, 10))
# disp.plot(cmap=plt.cm.Blues, ax=ax)
#
# # 设置坐标轴刻度（只显示5的倍数，字体加粗）
# ax.set_xticks(tick_indices)
# ax.set_xticklabels(tick_labels, fontdict=font_ticks, rotation=0)
#
# ax.set_yticks(tick_indices)
# ax.set_yticklabels(tick_labels, fontdict=font_ticks)
#
# # 设置图标题
# ax.set_title("Confusion Matrix", fontdict=font_title)
#
# # 设置x轴、y轴标题（加粗罗马字体）
# ax.set_xlabel("Predicted Label", fontdict=font_axis)
# ax.set_ylabel("True Label", fontdict=font_axis)
#
# plt.tight_layout()
# plt.show()

# 替换为你的混淆矩阵
# cm = confusion_matrix(all_labels, all_preds)
#
# # 分解为两个 20×20 的矩阵
# cm_part1 = cm[:20, :20]
# cm_part2 = cm[20:, 20:]
#
# # 标签
# labels1 = [str(i) for i in range(1, 21)]
# labels2 = [str(i) for i in range(21, 41)]
#
# # 提取5的倍数的坐标索引和标签
# tick_indices_20 = [i for i in range(20) if (i + 1) % 5 == 0]
# tick_labels_20_1 = [labels1[i] for i in tick_indices_20]
# tick_labels_20_2 = [labels2[i] for i in tick_indices_20]
#
# # 字体设置
# font_ticks = {'family': 'serif', 'name': 'Times New Roman', 'size': 16, 'weight': 'bold'}
# font_title = {'family': 'serif', 'name': 'Times New Roman', 'size': 18, 'weight': 'bold'}
# font_axis = {'family': 'serif', 'name': 'Times New Roman', 'size': 17, 'weight': 'bold'}
#
# # 绘制第一个 20×20 混淆矩阵
# fig1, ax1 = plt.subplots(figsize=(8, 8))
# disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_part1, display_labels=labels1)
# disp1.plot(cmap=plt.cm.Blues, ax=ax1, colorbar=False)
#
# ax1.set_xticks(tick_indices_20)
# ax1.set_xticklabels(tick_labels_20_1, fontdict=font_ticks, rotation=0)
# ax1.set_yticks(tick_indices_20)
# ax1.set_yticklabels(tick_labels_20_1, fontdict=font_ticks)
#
# ax1.set_title("Confusion Matrix (Class 1–20)", fontdict=font_title)
# ax1.set_xlabel("Predicted Label", fontdict=font_axis)
# ax1.set_ylabel("True Label", fontdict=font_axis)
#
# plt.tight_layout()
#
# # 绘制第二个 20×20 混淆矩阵
# fig2, ax2 = plt.subplots(figsize=(8, 8))
# disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_part2, display_labels=labels2)
# disp2.plot(cmap=plt.cm.Blues, ax=ax2, colorbar=False)
#
# ax2.set_xticks(tick_indices_20)
# ax2.set_xticklabels(tick_labels_20_2, fontdict=font_ticks, rotation=0)
# ax2.set_yticks(tick_indices_20)
# ax2.set_yticklabels(tick_labels_20_2, fontdict=font_ticks)
#
# ax2.set_title("Confusion Matrix (Class 21–40)", fontdict=font_title)
# ax2.set_xlabel("Predicted Label", fontdict=font_axis)
# ax2.set_ylabel("True Label", fontdict=font_axis)
#
# plt.tight_layout()
# plt.show()




# Grad-CAM 类定义
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self._register_hooks()
#
#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output.detach()
#
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].detach()
#
#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_backward_hook(backward_hook)
#
#     def generate(self, input_tensor, target_class=None):
#         self.model.zero_grad()
#         output = self.model(input_tensor)
#         if target_class is None:
#             target_class = output.argmax(dim=1).item()
#         loss = output[:, target_class]
#         loss.backward()
#         pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
#         activations = self.activations[0]
#         for i in range(activations.shape[0]):
#             activations[i, :, :] *= pooled_gradients[i]
#         heatmap = torch.mean(activations, dim=0).cpu().numpy()
#         heatmap = np.maximum(heatmap, 0)
#         heatmap /= np.max(heatmap) + 1e-6
#         return heatmap
#
#
# def show_cam_on_image(img, heatmap, alpha=0.5):
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = heatmap_color * alpha + np.uint8(img * 255)
#     superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
#     return superimposed_img
#
# # Grad-CAM 可视化
# grad_cam = GradCAM(model, model.conv5)
#
# # 创建保存文件夹
# output_dir = "gradcam_images"
# os.makedirs(output_dir, exist_ok=True)
#
# # 从 test_dataset 中随机选择 100 张图像索引
# indices = random.sample(range(len(test_dataset)), 300)
#
# print("Saving Grad-CAM for 100 test images...")
#
# for count, idx in enumerate(indices):
#     image, label = test_dataset[idx]
#     input_tensor = image.unsqueeze(0).to(device)
#     label_tensor = label.to(device)
#
#     pred = model(input_tensor).argmax(dim=1).item()
#     true = torch.argmax(label_tensor).item()
#     heatmap = grad_cam.generate(input_tensor, pred)
#
#     img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
#     superimposed_img = show_cam_on_image(img_np, heatmap)
#
#     label_index = torch.argmax(label_tensor).item() + 1  # 真实标签编号，从1开始
#     if f"label_{label_index}" not in locals():
#         locals()[f"label_{label_index}"] = 0
#     locals()[f"label_{label_index}"] += 1
#     img_id = locals()[f"label_{label_index}"]
#
#     filename = f"label_{label_index:02d}_{img_id:03d}.jpg"
#     cv2.imwrite(os.path.join(output_dir, filename), superimposed_img)
