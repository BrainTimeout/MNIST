#include "mnist.h"

Mnist::Mnist() {
    // 从资源文件中加载模型
    QFile modelFile(":/lenet_traced.pt"); // 资源路径
    if (!modelFile.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("无法打开模型文件！");
    }

    // 将资源文件写入临时文件
    QTemporaryFile tempFile;
    if (!tempFile.open()) {
        throw std::runtime_error("无法创建临时文件！");
    }

    tempFile.write(modelFile.readAll());
    tempFile.close();

    // 使用 LibTorch 加载临时文件
    try {
        module = torch::jit::load(tempFile.fileName().toStdString());
        qDebug() << "模型加载成功！";
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("模型加载失败: " + std::string(e.what()));
    }
}

int Mnist::predict(const QImage &image) {
    // 将QImage转换为灰度图像，并且缩放到28x28
    QImage convertedImage = image.convertToFormat(QImage::Format_Grayscale8);
    convertedImage = convertedImage.scaled(28, 28, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);

    // 创建一个Tensor，大小为1x1x28x28（MNIST图像大小28x28）
    at::Tensor tensor = torch::zeros({1, 1, 28, 28}, at::kFloat);

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            int pixelValue = qGray(convertedImage.pixel(j, i));  // 获取灰度值
            if(pixelValue == 255) pixelValue = 0; //背景从白色变为透明
            tensor[0][0][i][j] = static_cast<float>(pixelValue) / 255.0f;  // 归一化到[0, 1]
        }
    }

    // 使用模型进行预测
    torch::Tensor output = module.forward({tensor}).toTensor();

    // 获取最大概率对应的类别（假设是分类问题）
    int predictedClass = output.argmax(1).item<int>();

    return predictedClass;  // 返回预测类别
}

