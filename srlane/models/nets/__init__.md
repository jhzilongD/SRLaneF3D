# Nets 模块初始化文档

## 文件概述
`__init__.py` 是 nets 包的初始化文件，负责导入并暴露网络检测器模块。

## 代码块详解

### 1. 导入语句（第 3 行）
```python
from .detector import TwoStageDetector
```
**功能**：从 detector.py 模块中导入 TwoStageDetector 类，使其可以通过 `srlane.models.nets.TwoStageDetector` 访问。

## 整体功能总结

该初始化文件的作用是：
1. **模块暴露**：将 TwoStageDetector 类暴露给上层包，方便导入和使用
2. **包结构组织**：维护 nets 包的清晰结构
3. **统一入口**：提供统一的检测器访问入口

在 SRLane 项目中，用户可以通过 `from srlane.models.nets import TwoStageDetector` 来导入核心检测器。TwoStageDetector 是整个检测系统的顶层架构，整合了所有检测组件。