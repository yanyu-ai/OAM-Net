import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
from model import vit_base_patch16_8
from dataset_paths import TRAIN_PATH, TEST_PATH
from torchsummary import summary

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, paths, img_size, transform=None):
        self.images, self.labels = self.load_data(paths, img_size)
        self.transform = transform

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
        if self.transform:
            image = self.transform(image)
        return image, label

IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
])

train_dataset = CustomDataset(TRAIN_PATH, IMG_SIZE, transform=train_transform)
test_dataset = CustomDataset(TEST_PATH, IMG_SIZE)

# 将训练集划分为训练集和验证集（8:2比例）
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义训练和评估函数
def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_fn):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    print(f'Epoch [{epoch}], Loss: {avg_loss}, Accuracy: {accuracy}')
    return avg_loss, accuracy

def evaluate(model, data_loader, device, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    print(f'Evaluation Loss: {avg_loss}, Accuracy: {accuracy}')
    return avg_loss, accuracy, all_labels, all_preds

# 设置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 15
model = vit_base_patch16_8(num_classes=num_classes).to(device)  # 使用简化的模型

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 打开文件以保存损失和准确率
log_file = open("transformer.txt", "w")
log_file.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")

# 训练模型
num_epochs = 40
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, optimizer, train_loader, device, epoch, loss_fn)
    val_loss, val_accuracy, _, _ = evaluate(model, val_loader, device, loss_fn)

    # 将每个epoch的损失和准确率写入文件
    log_file.write(f"{epoch + 1},{train_loss},{train_accuracy},{val_loss},{val_accuracy}\n")

end_time = time.time()
log_file.close()

print(f"Total Training Time: {end_time - start_time:.2f} seconds")

# 测试模型
print("Final Test Evaluation")
_, _, all_labels, all_preds = evaluate(model, test_loader, device, loss_fn)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

# 绘制混淆矩阵
# cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BS", "HLG", "LG"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()
