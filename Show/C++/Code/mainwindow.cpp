#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->pen,&QRadioButton::clicked, ui->mouseTrackerWindow,&MouseTrackerWindow::setBrushBlack);
    connect(ui->eraser,&QRadioButton::clicked, ui->mouseTrackerWindow,&MouseTrackerWindow::setBrushWhite);
    connect(ui->lineWidthEdit,&LineWidthEdit::lineWidthChanged,ui->mouseTrackerWindow,&MouseTrackerWindow::setLineWidth);

    connect(ui->clearButton,&QAbstractButton::pressed,ui->mouseTrackerWindow,&MouseTrackerWindow::clearCanvas);

    connect(ui->predictButton, &QPushButton::clicked, this, [=]() {
        QImage image = ui->mouseTrackerWindow->getImage();
        int result = mnist.predict(image);
        ui->lcdnumber->display(result);
    });

    ui->pen->setChecked(true);
}

MainWindow::~MainWindow()
{
    delete ui;
}

