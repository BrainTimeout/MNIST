## 将PyTorch 模型转换为 TorchScript

`TorchScript` 是 PyTorch 提供的一种将 PyTorch 模型转换为可序列化和优化的格式的工具。它的主要目的是让 PyTorch 模型能够在没有 Python 环境的情况下运行。将模型转换为 TorchScript 形式，可以使用 `torch.jit.script` 或 `torch.jit.trace`。`torch.jit.script` 适用于动态模型（即模型中有控制流），而 `torch.jit.trace` 适用于静态模型（即模型中没有控制流）。在本次项目中，`LeNet` 模型是静态的，因此可以使用 `torch.jit.trace`。

进行处理的`python`代码如下:

```
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# 定义 LeNet 网络
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型
model = LeNet(num_classes=10)

# 在 CPU 上使用模型
device = torch.device("cpu")  # 强制使用 CPU
model.to(device)

# 加载保存的模型
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.eval()  # 设置为评估模式

# 数据预处理
transform = transforms.Compose([
    transforms.Lambda(lambda x: np.copy(x)),
    transforms.ToTensor(),
    transforms.Resize(28),
])

# 创建一个示例输入张量
example_input = torch.rand(1, 1, 28, 28).to(device)

# 使用 torch.jit.trace 将模型转换为 TorchScript
traced_model = torch.jit.trace(model, example_input)

# 保存 TorchScript 模型
traced_model.save("lenet_traced.pt")

print("模型已成功转换为 TorchScript 并保存为 'lenet_traced.pt'")
```

