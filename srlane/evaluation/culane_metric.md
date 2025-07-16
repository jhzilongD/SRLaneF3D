# SRLane CULane 评估指标

## 文件概述
该文件实现了车道线检测的官方 CULane 评估指标。它提供离散（官方）和连续IoU计算方法、车道线插值工具以及测量车道线检测性能的精确率、召回率和F1分数的综合评估函数。

## 代码结构

### 车道线可视化工具

#### 车道线绘制函数
```python
def draw_lane(lane, img=None, img_shape=None, width=30):
```

**作用**: 在二值面罩上将车道线坐标渲染为粗线，用于IoU计算。

**参数**:
- **`lane`** (np.ndarray): 车道线坐标，作为 `(N, 2)` 数组的 `(x, y)` 点
- **`img`** (np.ndarray, 可选): 要绘制的现有图像
- **`img_shape`** (tuple, 可选): 如果 `img` 为 None，新图像的形状
- **`width`** (int): Line thickness for lane rendering (default: 30 pixels)

**Implementation**:
- Creates binary mask if no image provided
- Draws thick lines between consecutive lane points using `cv2.line()`
- Uses 255 as line color for binary mask generation

**Returns**: Binary mask with lane rendered as thick white lines

### IoU Calculation Methods

#### Discrete Cross IoU
```python
def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640)):
```

**Purpose**: Computes IoU using pixel-based binary masks (official CULane method).

**Algorithm**:
1. **Lane Rendering**: Convert each lane to binary mask using `draw_lane()`
2. **Pairwise IoU**: Calculate IoU between all prediction-groundtruth pairs
3. **Binary Operations**: Use bitwise AND/OR for intersection/union calculation

**Mathematical Formula**:
```python
IoU = |intersection| / |union| = |pred & gt| / |pred | gt|
```

**Tensor Shapes**:
- **Input lanes**: List of `(N_i, 2)` arrays (variable length lanes)
- **Binary masks**: `(H, W)` for each lane
- **Output**: `(n_pred, n_gt)` IoU matrix

#### Continuous Cross IoU  
```python
def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
```

**Purpose**: Computes IoU using geometric shapes (alternative to discrete method).

**Algorithm**:
1. **Shape Creation**: Convert lanes to LineString geometries with buffer
2. **Image Clipping**: Intersect buffered lanes with image polygon
3. **Geometric IoU**: Calculate area-based intersection over union

**Shape Parameters**:
- **Buffer Distance**: `width/2` pixels around lane centerline
- **Cap Style**: 1 (round caps)
- **Join Style**: 2 (round joins)

**Benefits**: More geometrically accurate than discrete method

### Lane Interpolation

#### Spline Interpolation Function
```python
def interp(points, n=50):
```

**Purpose**: Smoothly interpolates lane points using spline curves for accurate evaluation.

**Parameters**:
- **`points`** (List[Tuple]): Lane coordinate points
- **`n`** (int): Interpolation density factor (default: 50)

**Algorithm**:
1. **Coordinate Separation**: Extract x and y coordinates
2. **Spline Fitting**: Use scipy's `splprep()` with appropriate spline order
3. **Dense Sampling**: Generate `(len(u)-1)*n + 1` interpolated points
4. **Evaluation**: Use `splev()` to compute interpolated coordinates

**Spline Parameters**:
- **Smoothing**: `s=0` (no smoothing, exact interpolation)
- **Knot Parameters**: `t=n` 
- **Spline Order**: `k=min(3, len(points)-1)` (cubic when possible)

**Output**: Dense array of interpolated lane points for accurate IoU calculation

### Core Evaluation Function

#### CULane Metric Calculation
```python
def culane_metric(pred, anno, ori_img_shape, width=30, iou_thresholds=[0.5], 
                  official=True, img_shape=(590, 1640)):
```

**Purpose**: Computes precision, recall, and F1 metrics for a single image.

##### Input Parameters:
- **`pred`**: List of predicted lane coordinates
- **`anno`**: List of ground truth lane coordinates  
- **`ori_img_shape`**: Original image dimensions
- **`width`**: Lane width for IoU calculation (default: 30)
- **`iou_thresholds`**: List of IoU thresholds for evaluation
- **`official`**: Whether to use discrete (True) or continuous (False) IoU
- **`img_shape`**: Target evaluation image size

##### Processing Pipeline:
1. **Coordinate Scaling**: Resize lanes to evaluation image dimensions
2. **Interpolation**: Generate dense lane representations
3. **IoU Calculation**: Compute cross-IoU matrix between predictions and ground truth
4. **Optimal Matching**: Use Hungarian algorithm for optimal assignment
5. **Threshold Application**: Count matches above each IoU threshold

##### Coordinate Scaling:
```python
xy_factors = (img_shape[1] / ori_img_shape[1], img_shape[0] / ori_img_shape[0])
def resize_lane(lane):
    return [[x * xy_factors[0], y * xy_factors[1]] for x, y in lane]
```

##### Optimal Assignment:
```python
row_ind, col_ind = linear_sum_assignment(1 - ious)  # Hungarian algorithm
```

##### Metric Computation:
For each IoU threshold:
```python
tp = int((ious[row_ind, col_ind] > thr).sum())  # True positives
fp = len(pred) - tp                              # False positives  
fn = len(anno) - tp                              # False negatives
```

**Returns**: Dictionary mapping thresholds to `[tp, fp, fn]` counts

### Data Loading Utilities

#### Single Image Data Loading
```python
def load_culane_img_data(path):
```

**Purpose**: Loads lane annotations from a single `.lines.txt` file.

**File Format**: Space-separated coordinate pairs per line
```
x1 y1 x2 y2 x3 y3 ...
x1 y1 x2 y2 x3 y3 ...
```

**Processing**:
1. **File Reading**: Read all lines from annotation file
2. **Coordinate Parsing**: Split lines and convert to floats
3. **Pair Formation**: Group coordinates into `(x, y)` tuples
4. **Filtering**: Remove lanes with fewer than 2 points

#### Batch Data Loading
```python
def load_culane_data(data_dir, file_list_path, load_size=False):
```

**Purpose**: Loads lane data for all images in a dataset split.

**Parameters**:
- **`data_dir`**: Root directory containing lane annotation files
- **`file_list_path`**: Path to text file listing image names
- **`load_size`**: Whether to load original image dimensions

**Processing**:
1. **Path Construction**: Build annotation file paths from image list
2. **Batch Loading**: Load annotations for all images
3. **Size Loading**: Optionally load original image dimensions using PIL

**Returns**: 
- **`data`**: List of lane annotations per image
- **`ori_img_shape`**: List of original image dimensions (if requested)

### Comprehensive Evaluation Function

#### Dataset Evaluation
```python
def eval_predictions(pred_dir, anno_dir, list_path, img_shape=(590, 1640),
                     iou_thresholds=[0.5], width=30, is_curvelanes=False,
                     official=True, sequential=False):
```

**Purpose**: Evaluates predictions across an entire dataset split with comprehensive metrics.

##### Parameters:
- **`pred_dir`**: Directory containing prediction files
- **`anno_dir`**: Directory containing ground truth annotations
- **`list_path`**: File listing images to evaluate
- **`img_shape`**: Evaluation image dimensions
- **`iou_thresholds`**: List of IoU thresholds for evaluation
- **`width`**: Lane width for IoU calculation
- **`is_curvelanes`**: Whether to use original image sizes (for CurveLanes dataset)
- **`official`**: Use official discrete IoU calculation
- **`sequential`**: Process sequentially vs parallel

##### Processing Pipeline:
1. **Data Loading**: Load predictions and ground truth annotations
2. **Size Handling**: Use original sizes for CurveLanes, fixed size for CULane
3. **Parallel Processing**: Use multiprocessing for faster evaluation
4. **Metric Aggregation**: Sum true positives, false positives, false negatives
5. **Score Calculation**: Compute precision, recall, F1 for each threshold

##### Parallel Processing:
```python
from multiprocessing import Pool, cpu_count
with Pool(cpu_count()) as p:
    results = p.starmap(culane_metric, zip(predictions, annotations, ...))
```

##### Metric Calculation:
For each IoU threshold:
```python
precision = float(tp) / (tp + fp) if tp != 0 else 0
recall = float(tp) / (tp + fn) if tp != 0 else 0  
f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0
```

**Returns**: Dictionary with comprehensive evaluation results:
```python
{
    threshold: {
        "TP": true_positives,
        "FP": false_positives, 
        "FN": false_negatives,
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score
    }
}
```

## Data Flow and Metric Computation

### Evaluation Pipeline:
```
Predictions + Ground Truth → Coordinate Scaling → Interpolation → IoU Matrix → 
Optimal Matching → Threshold Application → Metric Calculation → Results
```

### Tensor Shape Evolution:
```
Raw Lanes: Variable length → Interpolated: (N, 50*n, 2) → 
Binary Masks: (H, W) → IoU Matrix: (n_pred, n_gt) → Metrics: Scalars
```

### Matching Algorithm:
1. **IoU Computation**: Calculate all pairwise IoUs
2. **Cost Matrix**: Use `1 - ious` as assignment cost
3. **Hungarian Algorithm**: Find optimal one-to-one matching
4. **Threshold Filtering**: Count matches above IoU threshold

## Configuration and Usage

### Basic Evaluation:
```python
results = eval_predictions(
    pred_dir="/path/to/predictions",
    anno_dir="/path/to/groundtruth", 
    list_path="/path/to/test_list.txt"
)
```

### Multi-Threshold Evaluation:
```python
results = eval_predictions(
    pred_dir=pred_dir,
    anno_dir=anno_dir,
    list_path=list_path,
    iou_thresholds=[0.3, 0.5, 0.7]
)
```

### Performance vs Accuracy Trade-offs:
- **Sequential Processing**: `sequential=True` for debugging
- **Parallel Processing**: `sequential=False` for speed (default)
- **Official IoU**: `official=True` for standard evaluation
- **Geometric IoU**: `official=False` for more accurate geometry

### Command Line Interface:
```python
def main():
    args = parse_args()
    for list_path in args.list:
        results = eval_predictions(args.pred_dir, args.anno_dir, list_path, ...)
```

## Integration with SRLane System

### Dataset Integration:
Used by `CULane` dataset class for evaluation:
```python
def evaluate(self, predictions, output_basedir):
    result = culane_metric.eval_predictions(
        output_basedir, self.data_root, self.list_path,
        iou_thresholds=[0.5], official=True
    )
    return result[0.5]["F1"]
```

### Model Evaluation:
```python
# During validation
f1_score = dataset.evaluate(model_predictions, output_dir)
logger.info(f"Validation F1: {f1_score:.4f}")
```

### Comprehensive Analysis:
For test evaluation with category breakdown:
```python
# Overall evaluation
results = eval_predictions(pred_dir, data_root, test_list)

# Category-specific evaluation  
for category, category_file in CATEGORYS.items():
    category_results = eval_predictions(pred_dir, data_root, category_file)
```

## Performance Considerations

### Computational Efficiency:
- **Parallel Processing**: Utilizes all CPU cores for faster evaluation
- **Optimized IoU**: Efficient binary operations for discrete IoU
- **Memory Management**: Processes images individually to control memory usage

### Accuracy vs Speed:
- **Interpolation Density**: Higher `n` values increase accuracy but slow evaluation
- **IoU Method**: Discrete IoU faster but less geometrically accurate
- **Lane Width**: Affects matching sensitivity and computational cost

This comprehensive evaluation module provides the gold standard for measuring lane detection performance on the CULane dataset, ensuring fair and accurate comparison of different methods while supporting both official evaluation protocols and research variations.