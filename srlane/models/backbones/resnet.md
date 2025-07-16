# ResNet Backbone 文档

## 文件概述
`resnet.py` 实现了 ResNet（残差网络）系列的骨干网络，用于从输入图像中提取多尺度特征。该模块提供了多种 ResNet 变体（ResNet-18/34/50/101/152）以及 ResNeXt 和 Wide ResNet 的实现。

## 代码块详解

### 1. 预训练模型 URL 字典（第 7-26 行）
```python
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    ...
}
```
**功能**：存储各种 ResNet 变体的预训练权重下载链接，这些权重在 ImageNet 数据集上预训练得到。

### 2. 辅助卷积函数（第 29-48 行）

#### conv3x3 函数
```python
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
```
**功能**：创建 3×3 卷积层，带有 padding 以保持特征图尺寸（当 stride=1 时）。
- `dilation`：空洞卷积参数，用于增大感受野
- `groups`：分组卷积参数，用于 ResNeXt 架构

#### conv1x1 函数
```python
def conv1x1(in_planes, out_planes, stride=1):
```
**功能**：创建 1×1 卷积层，主要用于通道数变换。

### 3. BasicBlock 类（第 50-97 行）
```python
class BasicBlock(nn.Module):
    expansion = 1
```
**功能**：ResNet 的基本残差块，用于 ResNet-18/34。

**结构**：
- 两个 3×3 卷积层
- 批归一化和 ReLU 激活
- 残差连接（跳跃连接）

**前向传播流程**：
1. 保存输入作为 identity
2. conv1 → bn1 → relu
3. conv2 → bn2
4. 如果需要下采样，对 identity 进行下采样
5. 将残差与 identity 相加
6. 最终 ReLU 激活

**张量形状变化**：
- 输入：[B, C_in, H, W]
- 输出：[B, C_out, H', W']（其中 C_out = planes，H' 和 W' 取决于 stride）

### 4. Bottleneck 类（第 99-147 行）
```python
class Bottleneck(nn.Module):
    expansion = 4
```
**功能**：ResNet 的瓶颈残差块，用于 ResNet-50/101/152。

**结构**：
- 1×1 卷积（降维）
- 3×3 卷积（特征提取）
- 1×1 卷积（升维）
- 残差连接

**特点**：
- `expansion = 4`：输出通道数是 planes 的 4 倍
- 支持分组卷积（ResNeXt）和空洞卷积

**张量形状变化**：
- 输入：[B, C_in, H, W]
- 中间（width）：[B, width, H, W]
- 输出：[B, planes × 4, H', W']

### 5. ResNetWrapper 类（第 149-187 行）
```python
@BACKBONES.register_module
class ResNetWrapper(nn.Module):
```
**功能**：ResNet 的包装类，用于适配 SRLane 框架的需求。

**主要参数**：
- `resnet`：ResNet 型号（如 "resnet18", "resnet50"）
- `pretrained`：是否加载预训练权重
- `replace_stride_with_dilation`：是否用空洞卷积替代步长下采样
- `out_conv`：是否添加输出卷积层
- `in_channels`：各层的输入通道数列表

**前向传播**：
1. 调用内部 ResNet 模型
2. 如果有输出卷积层，对最后一层特征进行变换
3. 返回多尺度特征列表

### 6. ResNet 主类（第 189-311 行）
```python
class ResNet(nn.Module):
```
**功能**：ResNet 的核心实现类。

**主要组件**：
- `conv1`：7×7 卷积层，步长为 2
- `bn1` + `relu`：批归一化和激活
- `maxpool`：最大池化，进一步下采样
- `layer1-4`：四个残差层组

**_make_layer 方法**（第 267-294 行）：
- 构建一个残差层组
- 第一个块可能进行下采样
- 支持空洞卷积替代步长下采样

**前向传播流程**：
1. 初始特征提取：conv1 → bn1 → relu → maxpool
2. 依次通过 layer1-4，收集每层输出
3. 返回多尺度特征列表

**张量形状变化**（以输入 [B, 3, H, W] 为例）：
- conv1 后：[B, 64, H/2, W/2]
- maxpool 后：[B, 64, H/4, W/4]
- layer1 后：[B, C1×expansion, H/4, W/4]
- layer2 后：[B, C2×expansion, H/8, W/8]
- layer3 后：[B, C3×expansion, H/16, W/16]
- layer4 后：[B, C4×expansion, H/32, W/32]

### 7. ResNet 构造函数（第 313-435 行）

包含多个预定义的 ResNet 变体构造函数：

- `resnet18/34`：使用 BasicBlock
- `resnet50/101/152`：使用 Bottleneck
- `resnext50_32x4d/resnext101_32x8d`：ResNeXt 变体，使用分组卷积
- `wide_resnet50_2/wide_resnet101_2`：Wide ResNet 变体，通道数加倍

## 整体功能总结

该模块的主要功能是：
1. **多尺度特征提取**：从输入图像提取不同分辨率的特征图
2. **预训练权重支持**：可加载 ImageNet 预训练权重，提升性能
3. **灵活的架构配置**：支持多种 ResNet 变体和自定义配置
4. **空洞卷积支持**：可通过空洞卷积保持特征图分辨率
5. **车道检测适配**：通过 ResNetWrapper 适配车道检测任务的特殊需求

在 SRLane 中，ResNet 作为骨干网络，负责从原始图像中提取丰富的多尺度特征表示，这些特征随后被送入后续的颈部网络和检测头进行车道线检测。