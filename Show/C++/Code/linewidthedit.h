#ifndef LINEWIDTHEDIT_H
#define LINEWIDTHEDIT_H

#include <QWidget>
#include <QSpinBox>
#include <QFrame>

QT_BEGIN_NAMESPACE
namespace Ui {
class LineWidthEdit;
}
QT_END_NAMESPACE

class LineWidthEdit : public QWidget
{
    Q_OBJECT

public:
    explicit LineWidthEdit(QWidget *parent = nullptr);
    ~LineWidthEdit();
    int getLineWidth() const;  // 获取当前的线条宽度

signals:
    void lineWidthChanged(int width);  // 当线条宽度改变时发射信号

private slots:
    void updateLineWidth(int width);  // 更新线条宽度

protected:
    void paintEvent(QPaintEvent *event) override;  // 绘制当前选定的线条

private:
    Ui::LineWidthEdit *ui;
    QSpinBox *lineWidthSpinBox;  // 用于设置线条宽度的 QSpinBox
    QFrame *lineWidthDisplayFrame;  // 显示线条宽度的 QFrame
    int lineWidth;  // 当前的线条宽度
};

#endif // LINEWIDTHEDIT_H
