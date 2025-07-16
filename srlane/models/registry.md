# Models Registry 模型注册器文档

## 文件概述
`registry.py` 实现了 SRLane 的模型组件注册和构建系统。它提供了一套完整的注册机制，允许各种模型组件（骨干网络、颈部网络、检测头、检测器）通过装饰器注册，并通过配置文件动态构建。

## 代码块详解

### 1. 导入模块（第 1-2 行）
```python
import torch.nn as nn
from srlane.registry import Registry, build_from_cfg
```
**功能**：导入 PyTorch 神经网络模块和 SRLane 的注册系统基础组件。

### 2. 组件注册表定义（第 4-7 行）
```python
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
NECKS = Registry("necks")
NETS = Registry("nets")
```
**功能**：为不同类型的模型组件创建独立的注册表。

**注册表说明**：
- `BACKBONES`：骨干网络注册表（如 ResNet 系列）
- `HEADS`：检测头注册表（如 LocalAngleHead, CascadeRefineHead）
- `NECKS`：颈部网络注册表（如 ChannelMapper）
- `NETS`：完整网络注册表（如 TwoStageDetector）

### 3. 通用构建函数（第 10-17 行）
```python
def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)
```
**功能**：通用的模型组件构建函数，支持单个组件和组件序列的构建。

**构建逻辑**：
1. **列表配置**：如果配置是列表，构建多个组件并用 `nn.Sequential` 连接
2. **单一配置**：直接构建单个组件
3. **参数传递**：支持默认参数传递给构建函数

### 4. 专用构建函数（第 20-33 行）

#### 骨干网络构建（第 20-21 行）
```python
def build_backbones(cfg):
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))
```
**功能**：构建骨干网络，传递完整配置作为默认参数。

#### 颈部网络构建（第 24-25 行）
```python
def build_neck(cfg):
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))
```
**功能**：构建颈部网络，用于特征适配和通道统一。

#### 检测头构建（第 28-29 行）
```python
def build_head(split_cfg, cfg):
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))
```
**功能**：构建检测头，支持分离的配置和完整配置。

**参数说明**：
- `split_cfg`：检测头的专用配置
- `cfg`：完整的模型配置

#### 完整网络构建（第 32-33 行）
```python
def build_net(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))
```
**功能**：构建完整的检测网络。

## 整体功能总结

### 注册系统的核心优势

1. **解耦设计**：
   - 配置与实现分离
   - 通过字符串名称引用组件
   - 易于扩展和替换组件

2. **动态构建**：
   - 运行时根据配置构建模型
   - 支持不同的组件组合
   - 无需修改代码即可调整架构

3. **参数传递**：
   - 自动传递配置参数
   - 支持默认参数设置
   - 确保组件间配置一致性

### 使用示例

#### 1. 组件注册
```python
# 在各组件模块中使用装饰器注册
@BACKBONES.register_module
class ResNetWrapper(nn.Module):
    def __init__(self, cfg):
        # 实现代码

@HEADS.register_module
class LocalAngleHead(nn.Module):
    def __init__(self, cfg):
        # 实现代码
```

#### 2. 配置文件
```python
# 配置文件示例
backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    # 其他参数...
)

neck = dict(
    type='ChannelMapper',
    in_channels=[64, 128, 256, 512],
    out_channels=256,
    # 其他参数...
)

rpn_head = dict(
    type='LocalAngleHead',
    num_points=72,
    # 其他参数...
)

roi_head = dict(
    type='CascadeRefineHead',
    num_points=72,
    # 其他参数...
)

net = dict(
    type='TwoStageDetector',
)
```

#### 3. 模型构建
```python
# 在训练脚本中构建模型
backbone = build_backbones(cfg)
neck = build_neck(cfg)
rpn_head = build_head(cfg.rpn_head, cfg)
roi_head = build_head(cfg.roi_head, cfg)
model = build_net(cfg)
```

### 配置系统的层次结构

```
cfg (完整配置)
├── backbone (骨干网络配置)
├── neck (颈部网络配置)
├── rpn_head (RPN头配置)
├── roi_head (ROI头配置)
└── net (完整网络配置)
```

### 构建流程分析

1. **配置解析**：
   ```python
   type_name = cfg.pop('type')  # 获取组件类型
   ```

2. **组件查找**：
   ```python
   component_class = registry.get(type_name)  # 从注册表获取类
   ```

3. **参数合并**：
   ```python
   kwargs = {**default_args, **cfg}  # 合并默认参数和配置
   ```

4. **实例创建**：
   ```python
   instance = component_class(**kwargs)  # 创建组件实例
   ```

### 扩展新组件

要添加新的模型组件：

1. **实现组件类**：
   ```python
   @BACKBONES.register_module  # 或其他相应注册表
   class NewBackbone(nn.Module):
       def __init__(self, cfg):
           # 实现代码
   ```

2. **更新配置**：
   ```python
   backbone = dict(
       type='NewBackbone',
       # 组件特定参数
   )
   ```

3. **无需修改构建代码**：注册系统自动处理新组件

### 错误处理

注册系统提供的错误处理：
- **未注册组件**：清晰的错误提示
- **参数错误**：详细的参数验证信息
- **类型检查**：确保配置格式正确

### 性能考虑

1. **延迟加载**：组件仅在需要时构建
2. **缓存机制**：避免重复构建相同组件
3. **内存优化**：自动管理组件生命周期

SRLane 的注册系统为整个框架提供了灵活、可扩展的组件管理机制，使得模型架构的配置和调整变得简单高效。