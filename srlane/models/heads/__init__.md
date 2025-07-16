# Heads 模块初始化文档

## 文件概述
`__init__.py` 是 heads 包的初始化文件，负责导入并暴露检测头模块。

## 代码块详解

### 1. 导入语句（第 3-4 行）
```python
from .srlane_head import CascadeRefineHead
from .local_angle_head import LocalAngleHead
```
**功能**：导入两个核心检测头类：
- `CascadeRefineHead`：级联精化检测头，负责第二阶段的车道线精化
- `LocalAngleHead`：局部角度检测头，负责第一阶段的车道线提议生成

## 整体功能总结

该初始化文件的作用是：
1. **模块暴露**：将两个重要的检测头类暴露给上层包
2. **两阶段检测支持**：提供完整的两阶段检测架构的检测头组件
3. **统一入口**：为 SRLane 的两阶段检测提供统一的检测头访问入口

在 SRLane 项目中，这两个检测头分别负责：
- **LocalAngleHead**：RPN阶段，生成车道线初始提议
- **CascadeRefineHead**：精化阶段，对提议进行迭代优化和分类