import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import cv2
from tqdm import tqdm
import time
import random
from dataset_paths import TRAIN_PATH, TEST_PATH


# 设置随机种子
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

# TRAIN_PATH = ["data_milk/train/augment/BS_1/","data_milk/train/augment/BS_4/","data_milk/train/augment/HLG_22/",
#               "data_milk/train/augment/HLG_25/","data_milk/train/augment/LG_2/","data_milk/train/augment/LG_6"]
# TEST_PATH = ["data_milk/1_12000/BS_1/", "data_milk/1_12000/BS_4/", "data_milk/1_12000/HLG_22/",
#              "data_milk/1_12000/HLG_25/","data_milk/1_12000/LG_2","data_milk/1_12000/LG_6"]


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, paths, img_size):
        self.images, self.labels = self.load_data(paths, img_size)

    def load_data(self, paths, img_size):
        data_images = []
        data_labels = []
        for i, dir in enumerate(paths):
            path_dir = os.listdir(dir)
            for file in path_dir:
                img_dir = os.path.join(dir, file)
                img = cv2.resize(cv2.imread(img_dir), (img_size, img_size)) / 255.0
                data_images.append(img)
                data_labels.append(i)

        data_images = np.array(data_images).astype("float32")
        data_labels = np.array(data_labels).astype("int64")
        return data_images, data_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image).permute(2, 0, 1)
        label = torch.tensor(label)
        return image, label

# 创建数据集和数据加载器
dataset = CustomDataset(TRAIN_PATH, IMG_SIZE)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

test_dataset = CustomDataset(TEST_PATH, IMG_SIZE)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义MobileNetV2的瓶颈块
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor):
        super(Bottleneck, self).__init__()
        self.use_residual = in_channels == out_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion_factor)
        self.dwconv = nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size=3, stride=1, padding=1, groups=in_channels * expansion_factor, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels * expansion_factor)
        self.conv2 = nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x.clone()  # 克隆残差
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.dwconv(out)))
        out = self.bn3(self.conv2(out))
        if self.use_residual:
            out = out + residual.clone()  # 克隆残差以避免就地操作
        return out

# 定义MobileNetV2模型
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=15):
        super(MobileNetV2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            Bottleneck(32, 16, 1),
            Bottleneck(16, 24, 6),
            Bottleneck(24, 24, 6),
            Bottleneck(24, 32, 6),
            Bottleneck(32, 32, 6),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 超参数
LEARNING_RATE = 0.0002  # 学习率
TRAIN_EPOCH = 40  # 训练轮数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型、损失函数和优化器
model = MobileNetV2(num_classes=15).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_start_time = time.time()  # 记录训练开始时间
log_file = open("V2Net.txt", "w")
log_file.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")

# 训练模型
for epoch in range(TRAIN_EPOCH):
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        total_train_correct += (torch.argmax(outputs, 1) == labels).sum().item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / len(train_dataset)

    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            total_val_correct += (torch.argmax(outputs, 1) == labels).sum().item()

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

for images, labels in tqdm(test_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs, 1)
    total_correct += (preds == labels).sum().item()
    all_labels.extend(labels.cpu().numpy())
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
n_classes = 15
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14])
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14])

# 计算 AUC (ROC-AUC值)
auc = roc_auc_score(all_labels_bin, all_preds_bin, average="macro", multi_class="ovr")
print(f'AUC: {auc:.4f}')


#
# # 绘制混淆矩阵
# cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BS_1", "BS_4", "HLG_22", "HLG_25", "LG_2", "LG_6"])
#
# # 自定义字体
# font = {'fontname': 'Times New Roman', 'fontsize': 16}  # 设置字体和字体大小
#
# # 绘制混淆矩阵
# disp.plot(cmap=plt.cm.Blues)
# plt.title('V2 Net', fontdict=font)
#
# # 设置轴标签字体
# plt.xlabel('Predicted label', fontdict=font)
# plt.ylabel('True label', fontdict=font)
#
# # 调整 x 轴和 y 轴的刻度字体
# plt.xticks(fontsize=12, fontname='Times New Roman')
# plt.yticks(fontsize=12, fontname='Times New Roman')
#
# plt.show()
