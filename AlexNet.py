import time
import torch
from sklearn.preprocessing import label_binarize
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, \
    roc_auc_score
import random
from dataset_paths import TRAIN_PATH, TEST_PATH

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

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
                img = cv2.resize(cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE), (img_size, img_size)) / 255.0
                data_images.append(img)
                data_labels.append(i)  # 使用类别索引作为标签
        data_images = np.array(data_images).astype("float32")
        data_labels = np.array(data_labels).astype("int64")
        return data_images, data_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image).unsqueeze(0)  # 增加一个维度以匹配输入通道
        label = torch.tensor(label)
        return image, label


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),  # 输入通道为1，输出通道为96
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 15),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


IMG_SIZE = 227  # 图像尺寸
# TRAIN_PATH = ["data_milk/train/augment/BS_1/","data_milk/train/augment/BS_4/","data_milk/train/augment/HLG_22/",
#               "data_milk/train/augment/HLG_25/","data_milk/train/augment/LG_2/","data_milk/train/augment/LG_6"]
# TEST_PATH = ["data_milk/1_12000/BS_1/", "data_milk/1_12000/BS_4/", "data_milk/1_12000/HLG_22/",
#              "data_milk/1_12000/HLG_25/","data_milk/1_12000/LG_2","data_milk/1_12000/LG_6"]

dataset = CustomDataset(TRAIN_PATH, IMG_SIZE)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = CustomDataset(TEST_PATH, IMG_SIZE)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

# 创建保存训练信息的文件
log_file = "AlexNet.txt"
with open(log_file, "w") as f:
    f.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")

num_epochs = 40
start_time = time.time()  # 记录训练开始时间
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_correct = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_correct += (torch.argmax(outputs, 1) == labels).sum().item()

    train_accuracy = total_correct / len(train_dataset)
    train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (torch.argmax(outputs, 1) == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = val_correct / len(val_dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # 将结果写入文件
    with open(log_file, "a") as f:
        f.write(f"{epoch + 1},{train_loss:.4f},{train_accuracy:.4f},{val_loss:.4f},{val_accuracy:.4f}\n")


# 统计模型参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total model parameters: {total_params}')

end_time = time.time()  # 记录训练结束时间
total_time = end_time - start_time  # 计算总的训练时间
print(f'Total training time: {total_time:.2f} seconds')

model.eval()
test_correct = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_correct += (torch.argmax(outputs, 1) == labels).sum().item()
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = test_correct / len(test_dataset)
print(f'Test Accuracy: {test_accuracy:.4f}')

# 计算多分类的精确率、召回率、F1分数
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
print(f'Precision (macro): {precision:.4f}')
print(f'Recall (macro): {recall:.4f}')
print(f'F1 Score (macro): {f1:.4f}')

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

# 对于 AUC 计算，需要将标签进行二值化
n_classes = 15
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14])
all_preds_bin = label_binarize(all_preds, classes=[0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14])

# 计算 AUC (ROC-AUC值)
auc = roc_auc_score(all_labels_bin, all_preds_bin, average="macro", multi_class="ovr")
print(f'AUC: {auc:.4f}')

# 计算模型占用的内存（以MB为单位）
param_size_in_bytes = total_params * 4  # 每个参数占用4字节（float32）
param_size_in_mb = param_size_in_bytes / (1024 ** 2)  # 转换为MB
print(f"Model Memory Usage: {param_size_in_mb:.2f} MB")

# # 绘制混淆矩阵
# cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BS_1", "BS_4", "HLG_22", "HLG_25", "LG_2", "LG_6"])
#
# # 自定义字体
# font = {'fontname': 'Times New Roman', 'fontsize': 16}  # 设置字体和字体大小
#
# # 绘制混淆矩阵
# disp.plot(cmap=plt.cm.Blues)
# plt.title('AlexNet', fontdict=font)
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
