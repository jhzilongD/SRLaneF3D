# Focal Loss 损失函数文档

## 文件概述
`focal_loss.py` 实现了 Focal Loss 损失函数，这是一种专门用于解决类别不平衡问题的损失函数。在车道线检测中，由于大部分像素都是背景，正负样本严重不平衡，Focal Loss 通过重新加权机制来缓解这个问题。

## 代码块详解

### 1. 导入模块（第 1-8 行）
```python
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
# Source: https://github.com/kornia/kornia/blob/f4f70fefb63287f72bc80cd96df9c061b1cb60dd/kornia/losses/focal.py
```
**功能**：导入必要的 PyTorch 模块，代码基于 Kornia 库的实现。

### 2. SoftmaxFocalLoss 类（第 11-23 行）
```python
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)
```
**功能**：Softmax版本的Focal Loss实现。

**参数说明**：
- `gamma`：聚焦参数，控制困难样本的权重
- `ignore_lb`：忽略标签值，通常为255

#### 前向传播（第 17-23 行）
```python
def forward(self, logits, labels):
    scores = F.softmax(logits, dim=1)
    factor = torch.pow(1. - scores, self.gamma)
    log_score = F.log_softmax(logits, dim=1)
    log_score = factor * log_score
    loss = self.nll(log_score, labels)
    return loss
```
**流程**：
1. 计算 Softmax 概率
2. 计算聚焦因子 `(1-p)^γ`
3. 应用聚焦因子到对数概率
4. 使用 NLL Loss 计算最终损失

### 3. One-Hot 编码函数（第 26-73 行）
```python
def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
```
**功能**：将整数标签转换为 one-hot 编码格式。

**参数说明**：
- `labels`：整数标签张量，形状为 `(N, *)`
- `num_classes`：类别数量
- `device`：目标设备
- `dtype`：目标数据类型
- `eps`：小的常数，避免数值问题

#### 实现流程（第 58-73 行）
```python
if not torch.is_tensor(labels):
    raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")
if not labels.dtype == torch.int64:
    raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")
if num_classes < 1:
    raise ValueError(f"The number of classes must be bigger than one. Got: {num_classes}")

shape = labels.shape
one_hot = torch.zeros(shape[0], num_classes, *shape[1:], device=device, dtype=dtype)
return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
```
**功能**：
1. 参数类型和值验证
2. 创建零张量
3. 使用 `scatter_` 方法填充 one-hot 向量
4. 添加小常数避免数值问题

### 4. Focal Loss 核心函数（第 76-126 行）
```python
def focal_loss(input: torch.Tensor,
               target: torch.Tensor,
               alpha: float,
               gamma: float = 2.0,
               reduction: str = "none",
               eps: float = 1e-8) -> torch.Tensor:
```
**功能**：Focal Loss 的核心实现函数。

**参数说明**：
- `input`：预测logits，形状为 `(B, C, *)`
- `target`：目标标签
- `alpha`：平衡因子，用于调节正负样本权重
- `gamma`：聚焦参数，默认2.0
- `reduction`：损失归约方式（none/mean/sum）
- `eps`：数值稳定性常数

#### 输入验证（第 86-99 行）
```python
if not torch.is_tensor(input):
    raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
if not len(input.shape) >= 2:
    raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")
if input.size(0) != target.size(0):
    raise ValueError(f"Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).")
if not input.device == target.device:
    raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
```

#### Focal Loss 计算（第 101-126 行）
```python
# 计算 Softmax 概率
input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

# 转换为 one-hot 编码（如果需要）
if len(input.shape) - 1 == len(target.shape):
    target = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

# 计算 Focal Loss
weight = torch.pow(-input_soft + 1., gamma)  # (1-p)^γ 聚焦权重
focal = -alpha * weight * torch.log(input_soft)  # -α*(1-p)^γ*log(p)
loss_tmp = torch.sum(target * focal, dim=1)

# 应用归约操作
if reduction == "none":
    loss = loss_tmp
elif reduction == "mean":
    loss = torch.mean(loss_tmp)
elif reduction == "sum":
    loss = torch.sum(loss_tmp)
```

**Focal Loss 公式**：
```
FL(pt) = -α(1-pt)^γ * log(pt)
```
其中：
- `pt` 是预测概率
- `α` 是平衡因子
- `γ` 是聚焦参数

### 5. FocalLoss 类（第 129-143 行）
```python
class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: float,
                 gamma: float = 2.0,
                 reduction: str = "none"):
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)
```
**功能**：Focal Loss 的 PyTorch 模块封装，便于在神经网络中使用。

## 整体功能总结

### Focal Loss 的核心优势

1. **解决类别不平衡**：
   - α 参数平衡正负样本权重
   - 特别适用于前景样本稀少的场景

2. **关注困难样本**：
   - γ 参数降低易分类样本的权重
   - 让模型更关注困难样本的学习

3. **数值稳定性**：
   - 使用 eps 避免对数计算中的数值问题
   - 支持不同的归约方式

### Focal Loss 的数学原理

**标准交叉熵损失**：
```
CE(pt) = -log(pt)
```

**平衡交叉熵损失**：
```
CE(pt) = -α*log(pt)
```

**Focal Loss**：
```
FL(pt) = -α(1-pt)^γ*log(pt)
```

**关键特性**：
- 当 `pt → 1`（易分类样本）时，`(1-pt)^γ → 0`，损失权重降低
- 当 `pt → 0`（困难样本）时，`(1-pt)^γ → 1`，损失权重保持较高
- `γ = 0` 时退化为平衡交叉熵损失

### 在 SRLane 中的应用

在车道线检测中，Focal Loss 主要用于：

1. **分类任务**：
   - 区分前景车道线和背景
   - 解决车道线像素稀少的问题

2. **CascadeRefineHead 中的应用**：
   ```python
   self.cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
   ```
   - α=0.25：给正样本更高权重
   - γ=2.0：标准聚焦参数

3. **训练优化**：
   - 提高困难车道线的检测性能
   - 减少背景噪声的影响
   - 加速收敛过程

通过 Focal Loss，SRLane 能够更好地处理车道线检测中的类别不平衡问题，提高对细节和困难样本的检测性能。