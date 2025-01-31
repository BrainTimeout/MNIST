import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw
import tkinter as tk
import numpy as np


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


# 创建模型
model = LeNet(num_classes=10)

# 选择设备（CUDA 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载保存的模型
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))  # 加载模型权重
model.eval()  # 设置为评估模式

# 数据预处理，确保图像大小为 28x28
transform = transforms.Compose([
    transforms.Lambda(lambda x: np.copy(x)),  # 确保图像数据是可写的
    transforms.ToTensor(),
    transforms.Resize(28),  # 保证图像大小为28x28
])


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("实时绘制数字")

        # 设置画布
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()

        # 初始化图像
        self.image = Image.new('L', (280, 280), color=255)  # 创建白色背景的图像
        self.draw = ImageDraw.Draw(self.image)

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)

        # 按钮
        self.predict_button = tk.Button(root, text="预测", command=self.predict)
        self.predict_button.pack()

        self.refresh_button = tk.Button(root, text="刷新", command=self.refresh)
        self.refresh_button.pack()

        # 显示预测结果
        self.result_label = tk.Label(root, text="预测结果: ", font=("Arial", 16))
        self.result_label.pack()

    def paint(self, event):
        # 越大，画的矩形越大，越密集，插值后对应的像素点值越大
        x1, y1 = (event.x - 4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=25)
        self.draw.line([x1, y1, x2, y2], fill=0, width=25)  # 绘制到PIL图像

    def predict(self):
        # 将画布上的图像进行处理
        image_processed = self.image.convert('L')  # 确保是灰度图
        # image_processed = image_processed.point(lambda p: 0 if p == 255 else 255)  # 反转颜色（确保背景为0，数字为1）

        # 预处理图像并转换为 Tensor
        image_tensor = transform(image_processed)
        image_tensor = image_tensor.unsqueeze(0)  # 增加批次维度
        image_tensor = image_tensor.to(device)
        # 黑白反转（将小于0.5的值设为1，大于0.5的值设为0）
        image_tensor = 1 - image_tensor  # 对于[0, 1]区间的浮点值，做反转操作

        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)  # 获取预测的标签

        # 显示预测结果
        self.result_label.config(text=f"预测结果: {predicted.item()}")

        # 可选：显示输入模型的图像
        if True:
            image_to_show = image_tensor.squeeze(0).cpu().numpy()  # 去掉批次维度
            plt.imshow(image_to_show[0], cmap='gray')  # 显示 28x28 图像
            plt.title("Inverted Image (28x28)")
            plt.axis('off')  # 不显示坐标轴
            plt.show()

    def refresh(self):
        # 清空画布和图像
        self.canvas.delete("all")  # 删除画布上的所有图形
        self.image = Image.new('L', (280, 280), color=255)  # 重新创建白色背景图像
        self.draw = ImageDraw.Draw(self.image)  # 重新创建绘制对象
        self.result_label.config(text="预测结果: ")  # 清空预测结果


# 创建Tkinter窗口并运行应用
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
