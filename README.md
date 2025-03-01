# MNIST
CV里的HellWorld：MNIST，加上QT里的HellWorld：窗口和按钮

- 最开始懒得换了，使用了VGG19作为训练模型，模型参数足足有470MB

  ```
  # 定义VGG19模型
  class VGGnet19(nn.Module):
      def __init__(self, num_classes=10, with_pool=True):
          super(VGGnet19, self).__init__()
          self.num_classes = num_classes
  
          # 通过add_layers添加不同的卷积块
          self.features = nn.Sequential(
              self._add_layers(1, 64, 2),  # 输入通道改为1，适应灰度图像
              self._add_layers(64, 128, 2),
              self._add_layers(128, 256, 2),
              self._add_layers(256, 512, 2)
          )
  
          # 使用自适应池化
          self.avgPool = nn.AdaptiveAvgPool2d((7, 7))  # 保证输出大小为7x7
  
          # 分类器部分
          self.classifier = nn.Sequential(
              nn.Linear(512 * 7 * 7, 4096),
              nn.ReLU(),
              nn.Dropout(),
              nn.Linear(4096, 4096),
              nn.ReLU(),
              nn.Dropout(),
              nn.Linear(4096, num_classes)  # 输出10类
          )
  
      def _add_layers(self, input_channels, num_filters, num_groups):
          layers = []
          for _ in range(num_groups):
              layers.append(nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1))
              layers.append(nn.ReLU())
              layers.append(nn.BatchNorm2d(num_filters))  # 添加BatchNorm
              input_channels = num_filters
          return nn.Sequential(*layers)
  
      def forward(self, x):
          x = self.features(x)  # 特征提取部分
          x = self.avgPool(x)  # 自适应池化
          x = x.view(x.size(0), -1)  # 展平特征图
          x = self.classifier(x)  # 分类器部分
          return x
  
  ```

- 之后将模型换成LeNet-5，发现收敛效果相同的情况下模型参数为187KB，用VGG19处理MNIST，我特么真是个大聪明。。。

  ```
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
  
  ```

  
