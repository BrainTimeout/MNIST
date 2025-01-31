# Release.zip 文件上传说明

由于 GitHub 对单个文件大小有 100MB 的限制，`Release.zip` 文件被分割成多个较小的部分进行上传。请按照以下步骤下载并合并这些文件，以恢复原始文件。

## 步骤 1: 下载所有分割的文件

请确保您下载了所有分割的文件，文件名如下：
- `Release_part_aa`
- `Release_part_ab`
- `Release_part_ac`

## 步骤 2: 合并文件

在 Windows 系统上，您可以通过以下步骤合并这些分割的文件：

在这些文件的目录下使用使用 `PowerShell`运行以下命令来合并文件：

```
Copy-Item -Path "Release_part_*" -Destination "Release.zip" -Force
```

这将把所有的分割文件合并成一个名为 `Release.zip` 的文件。

## 步骤 3: 使用合并后的文件

一旦合并完成，您就可以像使用原始 `Release.zip` 文件一样使用恢复的文件。如果您需要提取其中的内容，可以使用适当的解压工具（如 WinRAR、7-Zip）来提取。