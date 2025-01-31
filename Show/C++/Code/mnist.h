#ifndef MNIST_H
#define MNIST_H

#include <torch/script.h>
#include <QCoreApplication>
#include <QFile>
#include <QTemporaryFile>
#include <QDebug>
#include <ATen/Tensor.h>
#include <QImage>

class Mnist
{
public:
    Mnist();
    int predict(const QImage &image);

private:
    torch::jit::script::Module module;
};

#endif // MNIST_H
