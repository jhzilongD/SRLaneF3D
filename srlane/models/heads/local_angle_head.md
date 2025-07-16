# LocalAngleHead 局部角度检测头文档

## 文件概述
`local_angle_head.py` 实现了 LocalAngleHead 类，这是 SRLane 两阶段检测架构中的第一阶段（RPN阶段）检测头。它通过预测局部角度图来生成车道线的初始提议，并可选地进行语义分割辅助训练。

## 代码块详解

### 1. 导入模块（第 1-11 行）
```python
import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from srlane.models.registry import HEADS
from srlane.models.losses.seg_loss import SegLoss
```
**功能**：导入必要的数学函数、类型注解、PyTorch 模块和自定义损失函数。

### 2. LocalAngleHead 类定义（第 13-156 行）

#### 类装饰器和初始化（第 13-62 行）
```python
@HEADS.register_module
class LocalAngleHead(nn.Module):
    def __init__(self,
                 num_points: int = 72,
                 in_channel: int = 64,
                 cfg=None):
```
**参数说明**：
- `num_points`：车道线表示的点数（默认72个点）
- `in_channel`：输入特征通道数
- `cfg`：模型配置对象

**初始化流程**：

1. **基础参数设置**（第 29-34 行）：
   ```python
   self.n_offsets = num_points
   self.img_w = cfg.img_w
   self.img_h = cfg.img_h
   self.aux_seg = self.cfg.get("seg_loss_weight", 0.) > 0.
   self.feat_h, self.feat_w = self.cfg.angle_map_size
   ```

2. **先验坐标注册**（第 36-47 行）：
   ```python
   self.register_buffer(name="prior_ys",
                        tensor=torch.linspace(0, self.feat_h,
                                              steps=self.n_offsets,
                                              dtype=torch.float32))
   grid_y, grid_x = torch.meshgrid(...)
   ```
   - 创建 Y 坐标先验（从0到feat_h的等间距点）
   - 创建网格坐标（用于后续几何变换）

3. **角度预测卷积层**（第 49-52 行）：
   ```python
   self.angle_conv = nn.ModuleList()
   for _ in range(self.cfg.n_fpn):
       self.angle_conv.append(nn.Conv2d(in_channel, 1, 1, 1, 0, bias=False))
   ```
   为每个FPN层创建1×1卷积，输出1通道角度图

4. **可选语义分割分支**（第 54-60 行）：
   ```python
   if self.aux_seg:
       num_classes = self.cfg.max_lanes + 1
       self.seg_conv = nn.ModuleList()
       self.seg_criterion = SegLoss(num_classes=num_classes)
   ```

### 3. 权重初始化（第 63-65 行）
```python
def init_weights(self):
    for m in self.angle_conv.parameters():
        nn.init.normal_(m, 0., 1e-3)
```
**功能**：使用小方差正态分布初始化角度预测卷积层权重。

### 4. 前向传播方法（第 67-117 行）

#### 方法签名和参数
```python
def forward(self, feats: List[Tensor]):
```
**参数**：`feats` - 多尺度特征图列表

#### 前向传播流程

1. **角度预测**（第 78-89 行）：
   ```python
   theta_list = []
   if not self.training:
       feats = feats[-1:]  # 测试时只使用最深层特征
   for i, feat in enumerate(feats, 1):
       theta = self.angle_conv[len(feats) - i](feat).sigmoid()
       theta_list.append(theta)
   ```
   - 训练时使用所有FPN层，测试时只用最深层
   - Sigmoid激活确保角度值在[0,1]范围内

2. **可选语义分割预测**（第 85-89 行）：
   ```python
   if self.aux_seg:
       seg_list = []
       for i, feat in enumerate(feats, 1):
           seg = self.seg_conv[len(feats) - i](feat)
           seg_list.append(seg)
   ```

3. **角度插值和后处理**（第 90-96 行）：
   ```python
   angle = F.interpolate(theta_list[-1],
                         size=[self.feat_h, self.feat_w],
                         mode="bilinear",
                         align_corners=True).squeeze(1)
   angle = angle.detach()
   angle.clamp_(min=0.05, max=0.95)  # 限制角度范围
   ```

4. **几何变换生成车道线提议**（第 98-112 行）：
   ```python
   k = (angle * math.pi).tan()  # 角度转正切值
   bs, h, w = angle.shape
   ws = ((self.prior_ys.view(1, 1, self.n_offsets)
          - grid[:, 1].view(1, h * w, 1)) / k.view(bs, h * w, 1)
         + grid[:, 0].view(1, h * w, 1))  # 计算x坐标
   ws = ws / w  # 归一化
   valid_mask = (0 <= ws) & (ws < 1)  # 有效性掩码
   _, indices = valid_mask.max(-1)
   start_y = indices / (self.n_offsets - 1)  # 起始y坐标
   ```

**张量形状变化**：
- 输入特征：列表，每个元素 [B, C, H_i, W_i]
- 角度预测：[B, 1, H_i, W_i] → [B, H, W]
- 车道线提议：[B, H×W, 2+2+num_points]（置信度+起始点+x坐标序列）

### 5. 损失计算方法（第 119-151 行）
```python
def loss(self,
         pred_angle: List[Tensor],
         pred_seg: List[Tensor],
         gt_angle: List[Tensor],
         gt_seg: List[Tensor],
         loss_weight: Tuple[float] = [0.2, 0.2, 1.],
         ignore_value: float = 0.):
```

#### 角度损失计算（第 138-143 行）：
```python
angle_loss = 0
for pred, target, weight in zip(pred_angle, gt_angle, loss_weight):
    valid_mask = target > ignore_value
    angle_loss = (angle_loss
                  + ((pred - target).abs() * valid_mask).sum()
                  / (valid_mask.sum() + 1e-4)) * weight
```
**功能**：计算预测角度与真实角度之间的L1损失，仅在有效像素上计算。

#### 可选分割损失（第 144-149 行）：
```python
if self.aux_seg:
    seg_loss = 0
    for pred, target, weight in zip(pred_seg, gt_seg, loss_weight):
        seg_loss = seg_loss + self.seg_criterion(pred, target) * weight
```

### 6. 参数统计方法（第 153-155 行）
```python
def __repr__(self):
    num_params = sum(map(lambda x: x.numel(), self.parameters()))
    return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"
```

## 整体功能总结

LocalAngleHead 的核心功能包括：

### 1. 角度预测机制
- 通过多尺度特征预测局部角度图
- 角度表示车道线在每个位置的局部方向
- 使用双线性插值统一角度图尺寸

### 2. 几何车道线生成
- 基于角度图和几何约束生成车道线提议
- 使用正切函数将角度转换为斜率
- 通过射线投影计算车道线x坐标序列

### 3. 多任务学习支持
- 主任务：角度预测用于车道线提议生成
- 辅助任务：语义分割增强特征学习

### 4. 损失函数设计
- L1角度损失：确保角度预测准确性
- 可选分割损失：提供额外监督信号
- 多尺度损失：平衡不同分辨率的贡献

### 在 SRLane 中的作用

作为两阶段检测的第一阶段（RPN），LocalAngleHead 负责：

1. **提议生成**：从角度图生成车道线初始提议
2. **特征提取**：提供后续精化阶段所需的初始车道线表示
3. **多尺度处理**：利用FPN特征提供不同尺度的角度预测
4. **几何约束**：通过角度约束确保车道线的合理性

输出的车道线提议包含：
- 前2维：占位符（后续阶段使用）
- 第3维：起始y坐标
- 第4维：占位符（后续阶段使用） 
- 后n维：归一化的x坐标序列

这些提议随后传递给 CascadeRefineHead 进行进一步精化和分类。