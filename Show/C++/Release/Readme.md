# `Release` 文件

由于 `GitHub` 对单个文件大小有 `100MB `的限制，`Release` 文件压缩后的`Release.zip` 文件被分割成多个较小的部分进行上传。`Release`文件夹中包含了一个完整的QT项目，且含有该项目运行所需要的驱动文件，可以直接在windows系统上运行。

## 文件分割: 将`Release.zip` 分割为多个文件

使用了`git bash`中的`split`指令进行分割，分割为`Release_part_aa`,`Release_part_ab`,`Release_part_ac`这三个文件

```
split -b 50M Release.zip Release_part_
```

## 文件合并: 将具有固定前缀`Release_part_`的多个文件进行合并

### 方法 1：使用 `git bash` 或 Linux 环境

在windows环境下的 `git bash` 或 Linux 终端中，可以使用 `cat` 指令来合并文件：

```
cat Release_part_* > Release.zip
```

### 方法 2：使用 `PowerShell`

在windows环境下的 `PowerShell` 中，可以使用以下命令来合并文件：

```
Get-Content -Path Release_part_aa, Release_part_ab, Release_part_ac -Encoding Byte -Raw | Set-Content -Path Release.zip -Encoding Byte
```

#### PS:为什么 `PowerShell` 的 `cat` 不行？

在 `PowerShell` 中，`cat` 是 `Get-Content` 的别名，主要用于读取文本文件。当使用 `cat` 或 `Get-Content` 直接合并二进制文件时，`PowerShell` 会默认以文本编码方式处理文件内容，这会导致二进制数据被错误地转换，从而损坏文件。

因此，在 `PowerShell` 中合并二进制文件时，必须显式指定 `-Encoding Byte` 参数，以确保文件以二进制模式读取和写入，避免数据损坏。

## `Mnist.exe`说明

将所有的分割文件合并成 `Release.zip` 文件后,解压可得到一个`Release`文件夹。点击`Release`文件夹中的`Mnist.exe`文件运行项目。该项目可以识别手写框中的数字并在右侧显示运行结果，由于训练模型时使用的缩放比限制，在该`exe`文件下手写时，在某些位置以及某些字体大小的情况下，识别的效果可能不甚理想(如将5识别为8)。
