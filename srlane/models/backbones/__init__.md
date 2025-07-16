# Backbones 模块初始化文档

## 文件概述
`__init__.py` 是 backbones 包的初始化文件，负责导入并暴露骨干网络模块。

## 代码块详解

### 1. 导入语句（第 3 行）
```python
from .resnet import ResNet
```
**功能**：从 resnet.py 模块中导入 ResNet 类，使其可以通过 `srlane.models.backbones.ResNet` 访问。

## 整体功能总结

该初始化文件的作用是：
1. **模块暴露**：将 ResNet 类暴露给上层包，方便导入和使用
2. **包结构组织**：维护 backbones 包的清晰结构
3. **统一入口**：提供统一的骨干网络访问入口

在 SRLane 项目中，用户可以通过 `from srlane.models.backbones import ResNet` 来导入 ResNet 骨干网络。