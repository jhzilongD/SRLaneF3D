# ChannelMapper 颈部网络文档

## 文件概述
`channel_mapper.py` 实现了 ChannelMapper 类，这是一个特征金字塔网络（Feature Pyramid Network, FPN）的简化版本。它主要负责调整骨干网络输出的多尺度特征的通道数，使其统一为相同的输出通道数，便于后续检测头的处理。

## 代码块详解

### 1. 导入模块（第 1-6 行）
```python
from typing import List
import torch.nn as nn
from torch import Tensor
from ..registry import NECKS
```
**功能**：导入必要的类型注解、PyTorch 神经网络模块、张量类型和注册器。

### 2. ChannelMapper 类定义（第 9-51 行）

#### 类装饰器和文档字符串（第 9-17 行）
```python
@NECKS.register_module
class ChannelMapper(nn.Module):
    """Channel Mapper to reduce/increase channels of backbone features."""
```
**功能**：
- 使用 `@NECKS.register_module` 装饰器注册到颈部网络注册表
- 继承自 `nn.Module`，是标准的 PyTorch 模块

### 3. 初始化方法（第 18-35 行）
```python
def __init__(self,
             in_channels: List[int],
             out_channels: int,
             num_outs: int = None,
             **kwargs):
```
**参数说明**：
- `in_channels`：输入特征的通道数列表，对应骨干网络不同层的输出通道数
- `out_channels`：统一的输出通道数
- `num_outs`：输出特征图的数量，默认为输入特征图数量

**初始化流程**：
1. **参数验证**（第 25-29 行）：
   - 确保 `in_channels` 是列表类型
   - 计算输入和输出特征图数量
   - 确保输出数量不超过输入数量

2. **创建横向卷积层**（第 31-34 行）：
   ```python
   self.lateral_convs = nn.ModuleList()
   for i in range(self.num_ins - self.num_outs, self.num_ins):
       l_conv = nn.Conv2d(in_channels[i], out_channels, 1, 1, 0)
       self.lateral_convs.append(l_conv)
   ```
   - 为每个需要的输入特征创建 1×1 卷积层
   - 卷积核大小为 1×1，步长为 1，无 padding
   - 用于通道数变换而不改变特征图尺寸

### 4. 前向传播方法（第 36-46 行）
```python
def forward(self, inputs: List[Tensor]):
```
**功能**：将多尺度输入特征统一通道数后输出。

**前向传播流程**：
1. **输入验证**（第 38 行）：
   ```python
   assert len(inputs) >= len(self.in_channels)
   ```
   确保输入特征数量不少于预期

2. **输入裁剪**（第 39-41 行）：
   ```python
   if len(inputs) > len(self.in_channels):
       for _ in range(len(inputs) - len(self.in_channels)):
           del inputs[0]
   ```
   如果输入特征过多，删除前面的低分辨率特征

3. **通道映射**（第 42-45 行）：
   ```python
   outs = [
       lateral_conv(inputs[i]) for i, lateral_conv in
       enumerate(self.lateral_convs)
   ]
   ```
   对每个输入特征应用对应的 1×1 卷积进行通道映射

**张量形状变化**：
- 输入：列表，每个元素形状为 [B, C_in[i], H_i, W_i]
- 输出：元组，每个元素形状为 [B, out_channels, H_i, W_i]

### 5. 参数统计方法（第 48-50 行）
```python
def __repr__(self):
    num_params = sum(map(lambda x: x.numel(), self.parameters()))
    return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
```
**功能**：返回模型参数数量的字符串表示，单位为千（K）。

## 整体功能总结

ChannelMapper 的主要功能包括：

1. **通道统一**：将骨干网络输出的不同通道数特征统一为相同的通道数
2. **多尺度保持**：保持不同分辨率特征图的空间尺寸不变
3. **轻量级设计**：仅使用 1×1 卷积，计算开销小
4. **灵活配置**：支持选择使用哪些层的特征

### 在 SRLane 中的作用

在 SRLane 的两阶段检测架构中，ChannelMapper 位于骨干网络和检测头之间：

1. **骨干网络输出**：ResNet 输出多尺度特征，通道数分别为 [64, 128, 256, 512]
2. **ChannelMapper 处理**：将这些特征统一为相同通道数（如 128 或 256）
3. **传递给检测头**：统一通道数的特征便于检测头进行后续处理

### 典型配置示例

```python
# 假设 ResNet-18 输出特征通道数为 [64, 128, 256, 512]
# 只使用后 3 层特征，统一为 256 通道
channel_mapper = ChannelMapper(
    in_channels=[128, 256, 512],  # 对应 layer2, layer3, layer4
    out_channels=256,
    num_outs=3
)
```

这样设计使得后续的检测头可以处理统一格式的多尺度特征，简化了网络架构的复杂性。