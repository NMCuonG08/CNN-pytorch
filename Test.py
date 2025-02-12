import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    device = torch.device("cuda")  # Chuyển sang GPU
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")  # Nếu không có GPU, sử dụng CPU
    print("Using CPU")
# Biến đổi ảnh thành tensor và chuẩn hóa về khoảng [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Tải tập dữ liệu MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)


# Tạo DataLoader để huấn luyện theo batch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Ảnh MNIST có kích thước 28x28
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 lớp đầu ra (0-9)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten ảnh thành vector
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5  # Số lần huấn luyện toàn bộ dataset

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradient

        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Tính loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Cập nhật trọng số

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")
correct = 0
total = 0
with torch.no_grad():

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Chọn lớp có xác suất cao nhất
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

# Lấy một batch hình ảnh từ test_loader
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)  # Chuyển lên GPU

# Dự đoán
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Hiển thị hình ảnh và dự đoán
imshow(torchvision.utils.make_grid(images.cpu()))  # Cần .cpu()
print('Predicted:', ' '.join(f'{predicted[j].item()}' for j in range(8)))
