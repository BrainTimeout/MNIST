#ifndef MOUSETRACKERWINDOW_H
#define MOUSETRACKERWINDOW_H

#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <QPixmap>
#include <QVector>

class MouseTrackerWindow : public QWidget
{
    Q_OBJECT

public:
    explicit MouseTrackerWindow(QWidget *parent = nullptr);
    void clearCanvas();                               // 清空画布
    QImage getImage();          // 输出当前画布

public slots:
    void setLineWidth(int lineWidth);   //设置线条的粗细
    void setBrushBlack();
    void setBrushWhite();

protected:
    void mouseMoveEvent(QMouseEvent *event) override;  // 处理鼠标移动事件
    void paintEvent(QPaintEvent *event) override;      // 绘制事件

private:
    QPixmap canvas;                                   // 缓存的画布
    int lineWidth;                                       // 线条粗细
    QBrush brush;                                     // 画刷


};

#endif // MOUSETRACKERWINDOW_H
