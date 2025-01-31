#include "MouseTrackerWindow.h"
#include <QPainter>
#include <QMouseEvent>
#include <QImage>

MouseTrackerWindow::MouseTrackerWindow(QWidget *parent)
    : QWidget(parent), lineWidth(25)  // 初始化 radius
{
    setWindowTitle("Mouse Tracker");
    setFixedSize(500, 500);

    // 初始化QPen和QBrush，避免每次重绘时都创建
    brush.setColor(Qt::black); // 设置填充颜色为黑色
    brush.setStyle(Qt::SolidPattern);  // 设置为实心填充

    // 创建QPixmap缓存画布
    canvas = QPixmap(size());
    canvas.fill(Qt::white);  // 设置初始为白色背景

    this->setMouseTracking(true);  // 开启鼠标移动事件捕获
}

void MouseTrackerWindow::mouseMoveEvent(QMouseEvent *event)
{
    // 捕获鼠标的移动事件，并存储位置
    if (event->buttons() & Qt::LeftButton) {
        QPoint point = event->pos();
        QPainter painter(&canvas);
        painter.setPen(Qt::NoPen);
        painter.setBrush(brush);
        int radius = lineWidth/2;  // 圆的半径
        painter.drawEllipse(point, radius, radius);  // 在画布上绘制圆

        // 直接更新显示，而不是全重绘
        update();  // 请求重绘，仅绘制新添加的内容
    }
}

void MouseTrackerWindow::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.drawPixmap(0, 0, canvas);  // 绘制缓存的画布
}

void MouseTrackerWindow::setLineWidth(int lineWidth)
{
    // 设置线宽，确保半径为正数
    if (lineWidth > 0) {
        this->lineWidth = lineWidth;
    }
}


void MouseTrackerWindow::clearCanvas()
{
    // 清空存储的鼠标轨迹，并重绘
    canvas.fill(Qt::white);  // 清空画布内容
    update();
}

QImage MouseTrackerWindow::getImage()
{
    // 将当前画布转换为QImage并返回
    QImage image(size(), QImage::Format_ARGB32);
    QPainter painter(&image);
    painter.fillRect(image.rect(), Qt::white);  // 背景填充为白色
    render(&painter);  // 渲染窗口内容到图片

    return image;  // 返回生成的图像
}

void MouseTrackerWindow::setBrushBlack(){
    brush.setColor(Qt::black); // 设置填充颜色为黑色
};
void MouseTrackerWindow::setBrushWhite(){
    brush.setColor(Qt::white); // 设置填充颜色为白色
};
