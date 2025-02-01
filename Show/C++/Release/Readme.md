# `Release` 文件

由于 `GitHub` 对单个文件大小有 `100MB `的限制，`Release` 文件压缩后的`Release.zip` 文件被分割成多个较小的部分进行上传。`Release`文件夹中包含了一个完整的QT项目，且含有该项目运行所需要的驱动文件，可以直接运行。

## 文件分割: 将`Release.zip` 分割为多个文件

使用了`git bash`中的`split`指令进行分割，分割为`Release_part_aa`,`Release_part_ab`,`Release_part_ac`这三个文件

```
split -b 50M Release.zip Release_part_
```

## 文件合并: 将具有固定前缀`Release_part_`的多个文件进行合并

使用了`git bash`中的`cat`指令来合并文件

```
cat Release_part_* > Release.zip
```

这将把所有的分割文件合并成 `Release.zip` 文件,之后解压即可。
## `Mnist.exe`说明

点击`Release`文件中的`Mnist.exe`文件运行项目。可以识别手写框中的数字并在右侧显示运行结果，由于训练模型时使用的缩放比限制，在该`exe`文件下手写时，在某些位置以及某些字体大小的情况下，识别的效果可能不甚理想(如将5识别为8)。
