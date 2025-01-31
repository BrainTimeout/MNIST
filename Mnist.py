import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import time

# 自定义MNIST数据集类
import gzip  # 如果文件是压缩的，可以用 gzip 打开


class MNISTDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        self.images = self._load_images(image_file)
        self.labels = self._load_labels(label_file)
        self.transform = transform

    def _load_images(self, image_file):
        with open(image_file, 'rb') as f:
            # 跳过前16个字节的魔数和元数据
            f.read(16)
            # 读取所有图像数据
            images = np.frombuffer(f.read(), dtype=np.uint8)
            num_images = images.shape[0] // 28 // 28  # 计算图像数量
            images = images.reshape(num_images, 28, 28)  # 重塑为 (num_images, 28, 28)
        return images

    def _load_labels(self, label_file):
        with open(label_file, 'rb') as f:
            # 跳过前8个字节的魔数和元数据
            f.read(8)
            # 读取所有标签数据
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)  # 返回数据集中的图像数量

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义 LeNet 网络
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 卷积层1：输入1通道，输出6个通道，卷积核大小5x5，步幅1
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2：输入6通道，输出16个通道，卷积核大小5x5，步幅1
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 全连接层1：16*4*4 -> 120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层2：120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3：84 -> num_classes
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 卷积层 + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 16 * 4 * 4)
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 数据预处理，确保图像大小为 28x28
transform = transforms.Compose([
    transforms.Lambda(lambda x: np.copy(x)),  # 确保图像数据是可写的
    transforms.ToTensor(),
    # transforms.ToTensor() 会将输入图像的像素值从 [0, 255] 范围转换到 [0, 1] 范围，并且它将图像从 NumPy 数组转化为 PyTorch Tensor。
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.6, 1.4)),
    # degrees:最大旋转角度
    # translate=(a, b)：允许图像在水平和垂直方向上最多平移a和b。
    # scale=(a, b)：图像可以在a到b倍之间缩放。
    transforms.Resize(28),  # 保证图像大小为28x28
])

# 加载训练和测试数据集
train_dataset = MNISTDataset(
    image_file='./Data/train-images.idx3-ubyte',
    label_file='./Data/train-labels.idx1-ubyte',
    transform=transform
)

test_dataset = MNISTDataset(
    image_file='./Data/t10k-images.idx3-ubyte',
    label_file='./Data/t10k-labels.idx1-ubyte',
    transform=transform
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# 创建模型
model = LeNet(num_classes=10)  # 假设10类（MNIST）

# 选择设备（CUDA 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))  # 加载模型权重

model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 迭代训练数据
    for images, labels in tqdm(train_loader, desc="Training Epoch", ncols=100):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total * 100
    return epoch_loss, epoch_accuracy


# 评估函数
def evaluate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(valid_loader)
    epoch_accuracy = correct / total * 100
    return epoch_loss, epoch_accuracy


if __name__ == '__main__':
    # 训练与评估过程
    best_accuracy = 0.0
    num_epochs = 45
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录每个epoch的开始时间
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # 训练
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # 验证
        valid_loss, valid_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.2f}%")

        # 保存最佳模型
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.\n")

    # 保存最终模型
    torch.save(model.state_dict(), "vgg19_mnist.pth")
    print("Model saved to vgg19_mnist.pth")
