#include "modecheck.h"
#include "ui_modecheck.h"

ModeCheck::ModeCheck(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ModeCheck)
{
    ui->setupUi(this);
    connect(ui->pen,)
}

ModeCheck::~ModeCheck()
{
    delete ui;
}
