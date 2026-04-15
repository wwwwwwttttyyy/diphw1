# 图像几何变换与控制点变形项目说明

本项目包含了基于 Python 的图像基础几何变换（缩放、旋转、平移、翻转）以及基于控制点引导的图像变形（如 MLS/RBF 算法）的实现，并提供了可视化的网页交互界面（Gradio）。

## 1. 环境依赖与库安装

运行本项目需要安装以下第三方的 Python 库：
- `opencv`：用于图像处理。
- `numpy`：用于矩阵计算和数值处理。
- `gradio`：用于搭建UI。

### 安装方法

**方法一：通过 requirements.txt 安装**
在终端或命令行中，切换到项目目录下，运行以下命令即可一键安装所有依赖：
```bash
pip install -r requirements.txt
```

**方法二：手动安装**
直接使用 pip 手动安装对应的库：
```bash
pip install opencv-python numpy gradio
```

---

## 2. 如何使用

本项目包含两个主要功能脚本，运行对应的脚本后会在终端输出一个本地网页链接（通常为 `http://127.0.0.1:7860`），在浏览器中打开该链接即可进行交互式操作。

### 功能一： Global Transformation
此功能通过滑动条实时控制图像的缩放、旋转、平移和水平翻转。

**运行命令：**
```bash
python run_global_transform.py
```


### 功能二：Point Guided Deformation
此功能允许用户通过点击图像上的特征点，指定点到点的移动趋势（源控制点提取与目标位置分配），从而实现局部图像的平滑形变（如把闭着的嘴巴拉开、改变面部表情等效果）。

**运行命令：**
```bash
python run_point_transform.py
```

**使用步骤：**
1. 在左侧的图片区域上传待处理的图片。
2. **打点操作**（交替点击进行标记）：
   - **第一次点击**：选取要移动的**原始点**（系统会用蓝色点记录）。
   - **第二次点击**：选取该原始点期望到达的**目标点**（系统会用红色点记录），此时会自动绘制一个连接起点和终点的绿色箭头，表示变形趋势。
   - 重复此“交替点击”的过程，以添加多个控制点对。
3. 调整参数（如有），随后单击页面下方或侧边的运行/应用按钮（例如 **Run** 按钮）。右侧窗口便会生成并展示经过控制点驱动变形后的最终图像。
4. 如果需要重新打点重置点位，可以直接重新长传图片，或者刷新网页重新开始。

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).
