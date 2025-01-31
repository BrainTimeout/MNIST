#include "linewidthedit.h"
#include "ui_linewidthedit.h"
#include <QVBoxLayout>
#include <QPainter>

LineWidthEdit::LineWidthEdit(QWidget *parent)
    : QWidget(parent), ui(new Ui::LineWidthEdit), lineWidth(25)  // 设置默认的线条宽度为 25
{
    ui->setupUi(this);  // 设置UI界面

    // 获取lineWidthDisplayFrame和lineWidthSpinBox控件，这些在.ui文件中定义
    lineWidthDisplayFrame = ui->lineWidthDisplayFrame;
    lineWidthSpinBox = ui->lineWidthSpinBox;

    // 设置默认线条宽度
    lineWidthSpinBox->setRange(1, 50);  // 设置线条宽度的范围
    lineWidthSpinBox->setValue(25);  // 默认值为 25

    // 连接信号与槽
    connect(lineWidthSpinBox, &QSpinBox::valueChanged,
            this, &LineWidthEdit::updateLineWidth);

    // 初始化显示框
    updateLineWidth(lineWidthSpinBox->value());
}

int LineWidthEdit::getLineWidth() const
{
    return lineWidth;  // 返回当前的线条宽度
}

void LineWidthEdit::updateLineWidth(int width)
{
    lineWidth = width;  // 更新线条宽度
    emit lineWidthChanged(width);  // 发射信号
    update();  // 更新显示框
}

void LineWidthEdit::paintEvent(QPaintEvent *)
{
    QPainter painter(this);  // 在当前 QWidget 上进行绘制
    painter.setBrush(QBrush(Qt::black));  // 使用黑色填充
    painter.setPen(Qt::NoPen);  // 不需要边框

    // 获取QFrame的绝对位置和大小
    QRect frameRect = lineWidthDisplayFrame->geometry();

    // 保证宽度和高度相等，确保frame是正方形
    int size = qMin(frameRect.width(), frameRect.height());  // 使用最小的尺寸作为边长
    lineWidthDisplayFrame->setFixedSize(size, size);  // 设置 QFrame 为正方形

    // 计算圆形的半径
    int radius = lineWidth / 2;

    // 直接使用矩形框的中心点作为圆心
    QPoint center = frameRect.center();

    // 绘制一个圆形点，半径为当前的线条宽度
    painter.drawEllipse(center, radius, radius);  // 绘制圆形，使用 lineWidth 作为宽高
}

LineWidthEdit::~LineWidthEdit()
{
    delete ui;
}
