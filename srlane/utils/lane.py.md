# SRLane 车道线表示和操作

## 文件概述

该模块定义了 `Lane` 类，为 SRLane 检测系统中的车道标记提供了复杂的表示。该类处理车道线插值、坐标变换，并提供对车道线点的迭代访问。它使用样条插值从离散检测点创建平滑、连续的车道线表示。

## 导入和依赖

```python
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
```

**依赖项**:
- **NumPy**: 核心数组操作和数学函数
- **SciPy**: 提供用于平滑曲线插值的 `InterpolatedUnivariateSpline`

## 核心类定义

### Lane 类

```python
class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
```

**Purpose**: Represents a lane as a continuous curve that can be queried at arbitrary y-coordinates.

**Parameters**:
- `points` (np.ndarray): Lane points with shape `[N, 2]` where each row is `[x, y]`
  - Represents discrete lane detections from the model
  - Typically in image coordinates (pixels)
- `invalid_value` (float): Value returned for queries outside the lane's valid range
  - Default: -2.0 (indicates invalid/missing lane point)
- `metadata` (dict): Optional additional information about the lane
  - Could include confidence scores, lane type, etc.

### Initialization Process

```python
def __init__(self, points=None, invalid_value=-2., metadata=None):
    super(Lane, self).__init__()
    self.curr_iter = 0
    self.points = points
    self.invalid_value = invalid_value
    self.function = InterpolatedUnivariateSpline(points[:, 1],
                                                 points[:, 0],
                                                 k=min(3, len(points) - 1))
    self.min_y = points[:, 1].min() - 0.01
    self.max_y = points[:, 1].max() + 0.01
    self.metadata = metadata or {}
```

**Initialization Steps**:
1. **Iterator Setup**: `curr_iter = 0` for iteration protocol
2. **Data Storage**: Store original points and configuration
3. **Spline Creation**: Creates interpolating spline with:
   - Y-coordinates as input (`points[:, 1]`)
   - X-coordinates as output (`points[:, 0]`)
   - Adaptive degree: `k = min(3, len(points) - 1)`
4. **Valid Range**: Establishes y-coordinate bounds with small margin (0.01)
5. **Metadata**: Store additional lane information

**Spline Interpolation Details**:
- Uses cubic splines when possible (k=3) for smooth curves
- Automatically reduces degree for insufficient points
- Y-to-X mapping enables querying x-coordinates for given y-values

### String Representation

```python
def __repr__(self):
    return "[Lane]\n" + str(self.points) + "\n[/Lane]"
```

**Purpose**: Provides readable string representation for debugging and logging.

**Output Format**:
```
[Lane]
[[x1 y1]
 [x2 y2]
 ...
 [xN yN]]
[/Lane]
```

### Lane Evaluation

```python
def __call__(self, lane_ys):
    lane_xs = self.function(lane_ys)
    lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
    return lane_xs
```

**Function**: `__call__` (makes Lane objects callable)

**Purpose**: Evaluates the lane at given y-coordinates using spline interpolation.

**Parameters**:
- `lane_ys` (np.ndarray): Y-coordinates where lane should be evaluated
  - Can be single value or array of coordinates
  - Typically in normalized coordinates [0, 1] or pixel coordinates

**Returns**:
- `lane_xs` (np.ndarray): Corresponding x-coordinates
  - Same shape as input `lane_ys`
  - Contains `invalid_value` for out-of-range queries

**Algorithm**:
1. **Spline Evaluation**: Use interpolation function to get x-coordinates
2. **Range Checking**: Identify queries outside valid y-range
3. **Invalid Masking**: Set out-of-range results to `invalid_value`

**Mathematical Foundation**:
- Cubic spline interpolation provides C² continuity
- Preserves monotonicity when lane points are well-ordered
- Handles extrapolation gracefully by marking as invalid

### Array Conversion

```python
def to_array(self, size):
    img_h, img_w = size
    sample_y = range(img_h - 1, 0, -20)
    ys = np.array(sample_y) / float(img_h)
    xs = self(ys)
    valid_mask = (xs >= 0) & (xs < 1)
    lane_xs = xs[valid_mask] * img_w
    lane_ys = ys[valid_mask] * img_h
    lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1)
    return lane
```

**Function**: `to_array`

**Purpose**: Converts lane to array format suitable for evaluation or visualization.

**Parameters**:
- `size` (tuple): Image dimensions `(height, width)` in pixels

**Returns**:
- `lane` (np.ndarray): Lane points with shape `[M, 2]` where M ≤ len(sample_y)
  - Each row contains `[x_pixel, y_pixel]`
  - Only includes valid points within image boundaries

**Algorithm Steps**:
1. **Sampling Strategy**: 
   ```python
   sample_y = range(img_h - 1, 0, -20)  # Every 20 pixels from bottom to top
   ```
2. **Normalization**: Convert pixel y-coordinates to [0, 1] range
3. **Lane Evaluation**: Use spline to get corresponding x-coordinates
4. **Validity Filtering**: Keep only points where `0 ≤ x < 1` (within image)
5. **Coordinate Conversion**: Transform back to pixel coordinates
6. **Array Formation**: Combine x and y coordinates into single array

**Sampling Pattern**:
- Samples every 20 pixels vertically
- Goes from bottom to top (decreasing y)
- Provides sufficient resolution for most applications
- Balances detail with computational efficiency

### Iterator Protocol

```python
def __iter__(self):
    return self

def __next__(self):
    if self.curr_iter < len(self.points):
        self.curr_iter += 1
        return self.points[self.curr_iter - 1]
    self.curr_iter = 0
    raise StopIteration
```

**Functions**: `__iter__` and `__next__`

**Purpose**: Enables iteration over lane points using Python's iterator protocol.

**Behavior**:
- Iterates through original detection points (not interpolated)
- Returns one point per iteration as `[x, y]` coordinate pair
- Resets iterator counter after completion
- Follows Python iterator conventions

**Usage Example**:
```python
lane = Lane(points)
for x, y in lane:
    print(f"Point: ({x}, {y})")
```

## Data Flow and Transformations

### Input Data Flow
```
Detection Points [N, 2] → Spline Interpolation → Continuous Function
```

### Query Data Flow
```
Y-coordinates → Spline Evaluation → X-coordinates → Range Validation → Results
```

### Array Conversion Flow
```
Image Size → Y-sampling → Spline Query → Validity Filter → Pixel Coordinates → Array [M, 2]
```

## Role in SRLane System

### Dataset Integration
- **Ground Truth**: Represents annotated lane markings
- **Predictions**: Stores model output in standardized format
- **Evaluation**: Enables consistent comparison between predictions and ground truth

### Model Pipeline
1. **Data Loading**: Convert dataset annotations to Lane objects
2. **Training**: Provide target representations for loss computation
3. **Inference**: Store and manipulate detection results
4. **Post-processing**: Enable coordinate transformations and filtering

### Coordinate Systems
- **Normalized Coordinates**: [0, 1] range for scale-invariant processing
- **Pixel Coordinates**: Absolute image coordinates for visualization
- **Network Coordinates**: Feature map coordinates for loss computation

## Usage Examples

### Basic Lane Creation and Query
```python
import numpy as np
from srlane.utils.lane import Lane

# Create lane from detection points
points = np.array([[100, 400], [120, 300], [140, 200], [160, 100]])
lane = Lane(points)

# Query lane at specific y-coordinates
y_coords = np.array([150, 250, 350])
x_coords = lane(y_coords)
print(f"X-coordinates: {x_coords}")

# Convert to evaluation format
eval_points = lane.to_array(size=(480, 640))  # 480h x 640w image
print(f"Evaluation points shape: {eval_points.shape}")
```

### Iteration and Processing
```python
# Iterate over original points
for i, (x, y) in enumerate(lane):
    print(f"Point {i}: ({x:.1f}, {y:.1f})")

# Check lane representation
print(lane)  # Shows formatted point list
```

### Integration with Dataset
```python
class LaneDataset:
    def process_annotation(self, ann_data):
        # Convert annotation to Lane objects
        lanes = []
        for lane_points in ann_data['lanes']:
            if len(lane_points) > 1:  # Need at least 2 points for spline
                lane = Lane(np.array(lane_points))
                lanes.append(lane)
        return lanes
```

## Technical Considerations

### Spline Interpolation Properties
- **Smoothness**: C² continuous curves (smooth derivatives)
- **Local Control**: Changes in one region don't affect distant regions
- **Interpolation**: Exactly passes through all input points
- **Degree Adaptation**: Automatically handles cases with few points

### Numerical Stability
- **Range Checking**: Prevents extrapolation errors
- **Margin Addition**: 0.01 margin prevents floating-point edge cases
- **Invalid Value**: Clear indication of out-of-range queries

### Memory Efficiency
- **Lazy Evaluation**: Spline computed once, queried multiple times
- **Sparse Sampling**: to_array uses efficient sampling strategy
- **Original Preservation**: Keeps original points for reference

### Error Handling
- **Insufficient Points**: Gracefully reduces spline degree
- **Invalid Queries**: Returns sentinel values rather than raising errors
- **Empty Lanes**: Handles edge cases in iteration and evaluation

## Performance Characteristics

- **Initialization**: O(N log N) for spline fitting
- **Query**: O(1) amortized for spline evaluation  
- **Array Conversion**: O(H/20) where H is image height
- **Memory**: O(N) for points storage plus spline coefficients

This Lane class provides a robust, efficient foundation for representing and manipulating lane markings throughout the SRLane detection system.