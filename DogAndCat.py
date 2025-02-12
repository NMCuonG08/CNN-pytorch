import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Định nghĩa transform cho ảnh (resize và chuẩn hóa)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize ảnh về 128x128
    transforms.ToTensor(),  # Chuyển thành tensor
    transforms.Normalize([0.5], [0.5])  # Chuẩn hóa về [-1, 1]
])

# Tải dataset (đặt đường dẫn chính xác)
train_dir = "./DataDogCat/train"
test_dir = "./DataDogCat/test"

train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Kiểm tra dữ liệu
classes = train_dataset.classes  # Danh sách nhãn ('cat', 'dog')
print("Classes:", classes)


# Định nghĩa mô hình CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 lớp đầu ra (mèo hoặc chó)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Khởi tạo mô hình
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")

# Đánh giá mô hình
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')


# Hiển thị dự đoán mẫu
import numpy as np


def imshow_with_labels(imgs, labels, preds, nrow=8):
    # Unnormalize ảnh
    imgs = imgs / 2 + 0.5
    npimg = imgs.numpy()

    # Hiển thị hình ảnh
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(np.transpose(npimg, (1, 2, 0)))  # Chuyển trục để hiển thị đúng
    ax.axis("off")

    # Hiển thị nhãn dự đoán trên ảnh
    for i in range(len(labels)):
        row = i // nrow
        col = i % nrow
        ax.text(col * 125 + 10, row * 125 + 20, f"GT: {labels[i]}\nP: {preds[i]}",
                fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.7))

    plt.show()

# Lấy một batch ảnh từ test_loader
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Dự đoán
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Hiển thị hình ảnh với nhãn đúng (GT) và nhãn dự đoán (P)
imshow_with_labels(torchvision.utils.make_grid(images.cpu(), nrow=8),
                   [classes[labels[j]] for j in range(len(labels))],
                   [classes[predicted[j]] for j in range(len(predicted))])