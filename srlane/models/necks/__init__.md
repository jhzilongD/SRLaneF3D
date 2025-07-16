# Necks 模块初始化文档

## 文件概述
`__init__.py` 是 necks 包的初始化文件，负责导入并暴露颈部网络模块。

## 代码块详解

### 1. 导入语句（第 3 行）
```python
from .channel_mapper import ChannelMapper
```
**功能**：从 channel_mapper.py 模块中导入 ChannelMapper 类，使其可以通过 `srlane.models.necks.ChannelMapper` 访问。

## 整体功能总结

该初始化文件的作用是：
1. **模块暴露**：将 ChannelMapper 类暴露给上层包，方便导入和使用
2. **包结构组织**：维护 necks 包的清晰结构
3. **统一入口**：提供统一的颈部网络访问入口

在 SRLane 项目中，用户可以通过 `from srlane.models.necks import ChannelMapper` 来导入颈部网络模块。颈部网络在整个检测架构中起到特征适配和通道统一的关键作用。