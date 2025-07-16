# TwoStageDetector 两阶段检测器文档

## 文件概述
`detector.py` 实现了 TwoStageDetector 类，这是 SRLane 的核心检测器架构。它整合了骨干网络、颈部网络、RPN头和ROI头，实现了完整的两阶段车道线检测流程。

## 代码块详解

### 1. 导入模块（第 1-5 行）
```python
import torch.nn as nn
from srlane.models.registry import NETS
from srlane.models.registry import build_backbones, build_head, build_neck
```
**功能**：导入神经网络模块和模型构建函数。

### 2. TwoStageDetector 类定义（第 7-65 行）

#### 类装饰器和初始化（第 7-22 行）
```python
@NETS.register_module
class TwoStageDetector(nn.Module):
    """Base class for two-stage detector.
    Usually includes backbone, neck, rpn head and roi head.
    """
    def __init__(self, cfg):
        super(TwoStageDetector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.neck = build_neck(cfg)
        self.rpn_head = build_head(cfg.rpn_head, cfg)
        self.roi_head = build_head(cfg.roi_head, cfg)
```
**功能**：构建两阶段检测器的各个组件。

**组件说明**：
- `backbone`：骨干网络（如 ResNet），负责特征提取
- `neck`：颈部网络（如 ChannelMapper），负责特征适配
- `rpn_head`：第一阶段检测头（LocalAngleHead），生成车道线提议
- `roi_head`：第二阶段检测头（CascadeRefineHead），精化车道线

### 3. 特征提取方法（第 23-27 行）
```python
def extract_feat(self, batch):
    feat = self.backbone(batch["img"] if isinstance(batch, dict) else batch)
    feat = self.neck(feat)
    return feat
```
**功能**：从输入图像中提取多尺度特征。

**流程**：
1. **输入处理**：支持字典或张量格式的输入
2. **骨干特征提取**：通过骨干网络提取多尺度特征
3. **颈部特征处理**：统一特征通道数

**张量形状变化**：
- 输入图像：[B, 3, H, W]
- 骨干输出：列表，每个元素 [B, C_i, H_i, W_i]
- 颈部输出：列表，每个元素 [B, C_out, H_i, W_i]

### 4. 测试前向传播（第 29-32 行）
```python
def _forward_test(self, batch):
    feat = self.extract_feat(batch)
    rpn_result_dict = self.rpn_head(feat)
    return self.roi_head(feat, **rpn_result_dict)
```
**功能**：测试时的前向传播，只进行推理不计算损失。

**流程**：
1. **特征提取**：获取多尺度特征
2. **RPN阶段**：生成车道线提议
3. **ROI阶段**：精化车道线并输出最终结果

### 5. 训练前向传播（第 34-51 行）
```python
def _forward_train(self, batch):
    feat = self.extract_feat(batch)
    loss_dic = dict()

    rpn_result_dict = self.rpn_head(feat)
    rpn_loss = self.rpn_head.loss(**rpn_result_dict, **batch)
    loss_dic.update(rpn_loss)
    
    roi_result_dict = self.roi_head(feat, **rpn_result_dict)
    roi_loss = self.roi_head.loss(roi_result_dict, batch=batch)
    loss_dic.update(roi_loss)

    for loss_k, loss_v in loss_dic.items():
        loss_dic[loss_k] = loss_v * self.cfg.get(f"{loss_k}_weight", 1.)
    all_loss = sum(loss_dic.values())
    loss_dic["loss"] = all_loss

    return {"loss": all_loss, "loss_status": loss_dic}
```
**功能**：训练时的前向传播，计算各阶段损失。

**损失计算流程**：
1. **RPN损失计算**：
   - 角度预测损失
   - 可选的分割损失
   
2. **ROI损失计算**：
   - 分类损失（Focal Loss）
   - 回归损失（L1 Loss）
   - 几何损失（Line IoU Loss）
   - 注意力损失

3. **损失加权**：
   ```python
   loss_dic[loss_k] = loss_v * self.cfg.get(f"{loss_k}_weight", 1.)
   ```
   根据配置文件中的权重对各项损失进行加权

4. **总损失计算**：所有损失项的加权和

### 6. 主前向传播方法（第 53-56 行）
```python
def forward(self, batch):
    if self.training:
        return self._forward_train(batch)
    return self._forward_test(batch)
```
**功能**：根据训练/测试模式选择不同的前向传播路径。

### 7. 模型信息输出（第 58-64 行）
```python
def __repr__(self):
    separator_info = "======== Param. Info. ========"
    num_params = sum(map(lambda x: x.numel(), self.parameters()))
    info = f"#Params of {self._get_name()}: "
    info += f"{num_params / 10 ** 6:<.2f}[M]"
    return '\n'.join([separator_info, repr(self.backbone), repr(self.neck),
                      repr(self.rpn_head), repr(self.roi_head), info])
```
**功能**：生成模型结构和参数信息的详细报告。

**输出示例**：
```
======== Param. Info. ========
#Params of ResNetWrapper: 11.69[M]
#Params of ChannelMapper: 0.20[K]
#Params of LocalAngleHead: 0.77[K]
#Params of CascadeRefineHead: 2.36[K]
#Params of TwoStageDetector: 14.25[M]
```

## 整体功能总结

### TwoStageDetector 的核心架构

TwoStageDetector 实现了经典的两阶段检测范式：

1. **第一阶段（RPN）**：
   - 使用 LocalAngleHead 生成车道线提议
   - 基于角度预测生成初始车道线
   - 提供粗糙但覆盖面广的检测结果

2. **第二阶段（ROI）**：
   - 使用 CascadeRefineHead 精化车道线
   - 通过多次迭代优化车道线质量
   - 输出高精度的最终检测结果

### 数据流分析

#### 训练时数据流
```
输入图像 [B,3,H,W]
    ↓
骨干网络 → 多尺度特征 [B,C₁,H₁,W₁], [B,C₂,H₂,W₂], ...
    ↓
颈部网络 → 统一特征 [B,C,H₁,W₁], [B,C,H₂,W₂], ...
    ↓
RPN头 → 车道线提议 [B,N,76] + RPN损失
    ↓
ROI头 → 精化结果 [B,M,76] + ROI损失
    ↓
总损失 = RPN损失 + ROI损失
```

#### 测试时数据流
```
输入图像 [B,3,H,W]
    ↓
特征提取 → 多尺度特征
    ↓
RPN头 → 车道线提议 [B,N,76]
    ↓
ROI头 → 最终检测结果 [B,M,76]
```

### 配置参数示例

```python
# 典型的检测器配置
net = dict(
    type='TwoStageDetector',
    backbone=dict(type='ResNetWrapper', ...),
    neck=dict(type='ChannelMapper', ...),
    rpn_head=dict(type='LocalAngleHead', ...),
    roi_head=dict(type='CascadeRefineHead', ...),
    # 损失权重配置
    angle_loss_weight=1.0,
    seg_loss_weight=0.2,
    cls_loss_weight=2.0,
    l1_loss_weight=1.0,
    iou_loss_weight=2.0,
    attn_loss_weight=0.5,
)
```

### 关键优势

1. **模块化设计**：各组件独立，易于替换和扩展
2. **两阶段优化**：先粗糙检测再精细优化，提高性能
3. **灵活配置**：支持各种损失权重和超参数调节
4. **端到端训练**：所有组件联合优化，性能最佳

### 在 SRLane 中的作用

TwoStageDetector 是 SRLane 的核心，它：
1. **统一架构**：整合所有检测组件
2. **训练管理**：协调各阶段的损失计算
3. **推理入口**：提供统一的检测接口
4. **性能保证**：通过两阶段设计确保检测精度

这种设计使得 SRLane 能够在保持高精度的同时，维持良好的可扩展性和可维护性。