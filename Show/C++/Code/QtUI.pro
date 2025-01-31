QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    linewidthedit.cpp \
    main.cpp \
    mainwindow.cpp \
    mnist.cpp \
    mousetrackerwindow.cpp

HEADERS += \
    linewidthedit.h \
    mainwindow.h \
    mnist.h \
    mousetrackerwindow.h

FORMS += \
    linewidthedit.ui \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

QMAKE_PROJECT_DEPTH = 0

# 添加 LibTorch 配置
LIBTORCH_PATH = D:/Development/libtorch-win-shared-with-deps-2.5.1+cpu/libtorch

# 添加 LibTorch 头文件路径
INCLUDEPATH += $$LIBTORCH_PATH/include
INCLUDEPATH += $$LIBTORCH_PATH/include/torch/csrc/api/include

# 添加 LibTorch 库文件路径
LIBS += -L$$LIBTORCH_PATH/lib

# 链接 LibTorch 库
LIBS += -ltorch -lc10 -ltorch_cpu

RESOURCES += \
    res.qrc
