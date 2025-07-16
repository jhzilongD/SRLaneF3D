# Models 模块初始化文档

## 文件概述
`__init__.py` 是 models 包的根初始化文件，负责导入并暴露整个模型系统的核心组件。

## 代码块详解

### 1. 子模块导入（第 2-5 行）
```python
from .backbones import *
from .heads import *
from .nets import *
from .necks import *
```
**功能**：使用通配符导入各个子模块的所有公开组件。

**导入的组件**：
- **backbones**：骨干网络（ResNet 系列）
- **heads**：检测头（LocalAngleHead, CascadeRefineHead）
- **nets**：完整检测器（TwoStageDetector）
- **necks**：颈部网络（ChannelMapper）

### 2. 构建函数导入（第 6 行）
```python
from .registry import build_backbones
```
**功能**：导入骨干网络的构建函数，提供统一的模型构建接口。

## 整体功能总结

该初始化文件的作用是：

### 1. 统一模块入口
通过根级初始化文件，用户可以直接从 `srlane.models` 导入所有模型组件：
```python
from srlane.models import ResNetWrapper, TwoStageDetector
from srlane.models import LocalAngleHead, CascadeRefineHead
from srlane.models import ChannelMapper
from srlane.models import build_backbones
```

### 2. 简化导入路径
无需记住复杂的子模块路径：
```python
# 简化前
from srlane.models.backbones.resnet import ResNetWrapper
from srlane.models.heads.local_angle_head import LocalAngleHead

# 简化后
from srlane.models import ResNetWrapper, LocalAngleHead
```

### 3. 保持向后兼容
通过统一入口，可以在不破坏现有代码的情况下重组内部模块结构。

### 4. 支持模块发现
IDE 和开发工具可以更容易地发现和自动补全可用的模型组件。

## SRLane 模型架构概览

通过此初始化文件暴露的完整模型系统包括：

### 核心检测器
- **TwoStageDetector**：两阶段检测器，整合所有组件

### 骨干网络
- **ResNetWrapper**：ResNet 骨干网络包装器
- **ResNet**：原始 ResNet 实现

### 检测头
- **LocalAngleHead**：第一阶段 RPN 头，生成车道线提议
- **CascadeRefineHead**：第二阶段精化头，迭代优化车道线
- **MultiSegmentAttention**：多段注意力机制

### 颈部网络
- **ChannelMapper**：特征通道映射器

### 构建工具
- **build_backbones**：骨干网络构建函数
- **各种注册表**：组件注册和管理系统

### 典型使用场景

```python
from srlane.models import TwoStageDetector, build_backbones

# 构建完整检测器
detector = TwoStageDetector(cfg)

# 或单独构建组件
backbone = build_backbones(cfg)
```

这种设计使得 SRLane 的模型系统既模块化又易于使用，为研究和应用提供了便利的接口。