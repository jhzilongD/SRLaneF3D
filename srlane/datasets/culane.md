# SRLane CULane 数据集实现文档

## 文件概述
本文件实现了 `CULane` 数据集类，用于处理车道线检测的 CULane 数据集格式。CULane 是一个大规模的具有挑战性的车道线检测数据集，包含多种场景，如正常、拥挤、高亮、阴影、无车道线、箭头、弯道、交叉和夜间等情况。

## 代码结构

### 数据集常量
```python
LIST_FILE = {
    "train": "list/train_gt.txt",
    "val": "list/val.txt", 
    "test": "list/test.txt",
}

CATEGORYS = {
    "normal": "list/test_split/test0_normal.txt",
    "crowd": "list/test_split/test1_crowd.txt",
    "hlight": "list/test_split/test2_hlight.txt",
    "shadow": "list/test_split/test3_shadow.txt",
    "noline": "list/test_split/test4_noline.txt",
    "arrow": "list/test_split/test5_arrow.txt",
    "curve": "list/test_split/test6_curve.txt",
    "cross": "list/test_split/test7_cross.txt",
    "night": "list/test_split/test8_night.txt",
}
```

#### 作用:
- **`LIST_FILE`**: 将数据集划分映射到对应的文件列表
- **`CATEGORYS`**: 定义不同的测试场景以进行全面评估

### 类定义
```python
@DATASETS.register_module
class CULane(BaseDataset):
```
继承自 `BaseDataset` 并在数据集注册器中注册，以便自动实例化。

### 构造函数
```python
def __init__(self, data_root, split, processes=None, cfg=None):
```

#### 初始化过程:
1. **父类初始化**: 调用 `BaseDataset.__init__()`
2. **路径设置**: 使用 `LIST_FILE` 映射构建列表文件路径
3. **标注加载**: 调用 `load_annotations()` 填充 `self.data_infos`
4. **采样点**: 设置用于评估的 `h_samples`（y坐标从270到590像素）

#### 关键属性:
- **`list_path`**: 数据集划分文件的路径
- **`split`**: 当前数据集划分（"train", "val", "test"）
- **`h_samples`**: 用于评估采样的归一化y坐标

### 标注加载
```python
def load_annotations(self, diff_thr=15):
```

#### 作用:
加载和处理数据集划分的所有标注，通过缓存机制优化性能。

#### 缓存机制:
- **缓存位置**: `.cache/culane_{split}.pkl`
- **缓存优势**: 显著加快后续数据集初始化速度
- **缓存失效**: 数据集变化时需要手动删除

#### 处理流程:
1. **文件解析**: 逐行读取数据集列表文件
2. **路径构建**: 为图像和标注构建绝对路径
3. **重复检测**: 对于训练集，使用像素差异阈值过滤近似重复图像
4. **标注处理**: 
   - 从 `.lines.txt` 文件加载车道线坐标
   - 过滤无效点（负坐标）
   - 移除车道线内的重复点
   - 过滤点数不足的车道线（< 3个点）
   - 按y坐标排序车道线点（从下到上）

#### 数据结构:
`self.data_infos` 中的每个条目包含:
```python
{
    "img_name": "relative/path/to/image.jpg",
    "img_path": "/absolute/path/to/image.jpg", 
    "mask_path": "/absolute/path/to/mask.png",  # 可选
    "lane_exist": np.array([0,1,1,0]),  # 车道线存在标志
    "lanes": [[(x1,y1), (x2,y2), ...], ...]  # 车道线坐标
}
```

#### 重复过滤逻辑:
- **阈值**: `diff_thr=15`（平均像素差异）
- **计算**: 所有像素和通道的平均绝对差异
- **目的**: 移除连续的相似帧以提高训练多样性

### 预测字符串生成
```python
def get_prediction_string(self, pred):
```

#### 作用:
将模型预测转换为 CULane 评估格式的字符串表示。

#### 输入格式:
- **`pred`**: 具有可调用接口 `lane(ys)` 的车道线对象列表，返回x坐标

#### 处理步骤:
1. **Y采样**: 使用 `self.h_samples` 进行一致的y坐标采样
2. **坐标转换**: 调用 `lane(ys)` 在采样点获取x坐标
3. **有效性过滤**: 移除图像边界外的点（`xs >= 0` 且 `xs < 1`）
4. **去归一化**: 将归一化坐标转换为像素坐标
5. **排序**: 反转坐标（从下到上到从上到下）
6. **字符串格式化**: 创建空格分隔的5位小数精度的坐标对

#### 输出格式:
```
x1.xxxxx y1.xxxxx x2.xxxxx y2.xxxxx ...
x1.xxxxx y1.xxxxx x2.xxxxx y2.xxxxx ...
```

### 评估方法
```python
def evaluate(self, predictions, output_basedir):
```

#### 作用:
使用官方 CULane 指标对模型预测与真实标签进行评估。

#### 评估流程:
1. **输出生成**: 
   - 创建与数据集布局匹配的目录结构
   - 将预测字符串写入 `.lines.txt` 文件
   - 保持原始文件名约定

2. **类别评估**（仅测试集）:
   - 分别评估每个场景类别
   - 提供每种驾驶条件下的详细性能分解

3. **整体评估**:
   - 计算整个数据集划分的指标
   - 使用 0.5 的 IoU 阈值进行车道线匹配
   - 返回 F1 分数作为主要指标

#### 评估指标:
- **IoU 阈值**: 0.5（CULane 标准）
- **主要指标**: F1 分数
- **附加指标**: 精确率、召回率、真正例、假正例、假负例

### 数据流和张量形状

#### 标注加载流程:
```
列表文件 → 解析行 → 提取路径 → 加载车道线坐标 → 过滤排序 → 缓存
```

#### 预测评估流程:
```
模型预测 → 坐标采样 → 验证 → 去归一化 → 字符串格式 → 文件输出 → 指标计算
```

#### 车道线坐标格式:
- **存储**: 每条车道线的 `(x, y)` 元组列表
- **坐标系**: 图像像素坐标
- **排序**: Y坐标按降序排列（从下到上）
- **过滤**: 移除无效坐标和重复点

## 配置依赖

### 必需参数:
- **`ori_img_h`**: 原始图像高度（CULane 通常为 590）
- **`ori_img_w`**: 原始图像宽度（CULane 通常为 1640）

### 数据集结构要求:
```
data_root/
├── list/
│   ├── train_gt.txt
│   ├── val.txt  
│   ├── test.txt
│   └── test_split/
│       ├── test0_normal.txt
│       ├── test1_crowd.txt
│       └── ...
├── driver_*/
│   ├── *.jpg (图像)
│   └── *.lines.txt (标注)
└── laneseg_label_w16/ (可选面罩)
```

## 使用示例

### 数据集创建:
```python
dataset = CULane(
    data_root="/path/to/culane",
    split="train",
    processes=augmentation_pipeline,
    cfg=config
)
```

### 评估使用:
```python
f1_score = dataset.evaluate(
    predictions=model_predictions,
    output_basedir="/path/to/output"
)
```

### 预测格式:
```python
# 每个预测应该是一个可调用的车道线对象
lane_prediction = lambda ys: np.array([...])  # 为给定y坐标返回x坐标
predictions = [lane_prediction1, lane_prediction2, ...]
```

## 与 SRLane 系统的集成

### 训练集成:
- 从 `BaseDataset` 继承数据加载和处理功能
- 提供 SRLane 模型期望格式的车道线标注
- 通过处理流程支持数据增强

### 评估集成:
- 实现官方 CULane 评估协议
- 提供类别级别的性能分析
- 与 SRLane 的验证流程集成

### 模型兼容性:
- 期望模型预测为可调用的车道线对象
- 处理坐标归一化和去归一化
- 通过预测接口支持各种车道线表示格式

## 性能考量

### 缓存策略:
- 标注加载缓存到磁盘以快速启动
- 缓存失效需要手动删除
- 对大型数据集显著加速

### 内存优化:
- 延迟加载图像（仅在访问时加载）
- 使用元组高效存储坐标
- 标注元数据的最小内存占用

### 评估效率:
- 指标计算中的并行处理
- 优化的坐标转换例程
- 预测文件的批处理

该实现为 CULane 数据集提供了健壮高效的接口，处理了数据集格式的复杂性，同时为 SRLane 训练和评估流程提供了清晰的集成。