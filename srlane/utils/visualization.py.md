# SRLane 可视化工具

## 文件概述

该模块为 SRLane 车道线检测系统提供可视化工具。它专注于使用 OpenCV 在图像上渲染车道线检测结果，支持多条车道线、可自定义颜色和灵活的输出选项。可视化系统对于调试、评估和展示车道线检测性能至关重要。

## 导入和依赖

```python
import os
import cv2
import os.path as osp
```

**依赖项**:
- **os**: 用于目录管理的文件系统操作
- **cv2**: 用于图像处理和渲染操作的 OpenCV
- **os.path**: 路径操作工具（别名为 `osp`）

## 颜色调色板定义

```python
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 255, 0),  # Lime
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (255, 0, 128),  # Pink
    (0, 128, 255),  # Light Blue
    (0, 255, 128),  # Spring Green
    (128, 255, 255),# Light Cyan
    (255, 128, 255),# Light Magenta
    (255, 255, 128),# Light Yellow
    (60, 180, 0),   # Dark Green
    (180, 60, 0),   # Brown
    (0, 60, 180),   # Dark Blue
    (0, 180, 60),   # Teal
    (60, 0, 180),   # Indigo
    (180, 0, 60),   # Maroon
    # Repeated colors for >21 lanes
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]
```

**Purpose**: Defines a comprehensive color palette for distinguishing multiple lanes in visualizations.

**Color Format**: BGR tuples `(B, G, R)` compatible with OpenCV conventions
- Note: OpenCV uses BGR instead of RGB color order

**Design Considerations**:
- **High Contrast**: Colors chosen for maximum visual distinction
- **Sufficient Variety**: 30+ colors to handle complex highway scenarios
- **Accessibility**: Includes bright, saturated colors for clear visibility
- **Cyclical**: Color list can be indexed with modulo for unlimited lanes

## Core Visualization Function

### Lane Rendering Function

```python
def imshow_lanes(img, lanes, show=False, out_file=None, width=4):
```

**Function**: `imshow_lanes`

**Purpose**: Renders lane detection results as connected line segments on an input image with customizable display and output options.

**Parameters**:
- `img` (np.ndarray): Input image array in BGR format
  - Shape: `[H, W, 3]` for color images
  - Modified in-place with lane overlay
- `lanes` (list): List of lane point sequences
  - Each lane: List of `(x, y)` coordinate tuples
  - Coordinates typically in pixel space
- `show` (bool): Whether to display image in OpenCV window
  - Default: False (no display)
  - True: Shows image until key press
- `out_file` (str, optional): Path to save rendered image
  - Default: None (no file output)
  - Creates directories if needed
- `width` (int): Line thickness for lane rendering
  - Default: 4 pixels
  - Higher values for better visibility in high-resolution images

**Returns**: None (modifies input image in-place)

### Implementation Details

#### Lane Point Processing

```python
lanes_xys = []
for _, lane in enumerate(lanes):
    xys = []
    for x, y in lane:
        if x <= 0 or y <= 0:
            continue
        x, y = int(x), int(y)
        xys.append((x, y))
    lanes_xys.append(xys)
```

**Processing Steps**:
1. **Point Validation**: Filters out invalid coordinates (`x <= 0` or `y <= 0`)
   - Handles missing or invalid lane points gracefully
   - Common in lane detection where some points may be undetected
2. **Type Conversion**: Converts coordinates to integers for OpenCV compatibility
3. **Structure Preservation**: Maintains lane grouping for proper rendering

**Data Flow**:
```
Input: [[(x1,y1), (x2,y2), ...], [(x3,y3), (x4,y4), ...], ...]
       ↓ Filter invalid points
       ↓ Convert to integers  
Output: [[(int_x1,int_y1), (int_x2,int_y2), ...], ...]
```

#### Lane Sorting

```python
lanes_xys.sort(key=lambda xys: xys[0][0])
```

**Purpose**: Sorts lanes by the x-coordinate of their first valid point.

**Benefits**:
- **Consistent Coloring**: Same lanes get same colors across frames
- **Left-to-Right Ordering**: Natural visual ordering matching driving perspective
- **Deterministic Rendering**: Reproducible visualization results

**Sorting Logic**: Uses the first point's x-coordinate as the sorting key
- Assumes first valid point is representative of lane position
- Works well for typical highway scenarios where lanes are roughly vertical

#### Line Rendering

```python
for idx, xys in enumerate(lanes_xys):
    for i in range(1, len(xys)):
        cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
```

**Rendering Process**:
1. **Color Assignment**: Uses lane index to select color from palette
   - `COLORS[idx]` with automatic wraparound for many lanes
2. **Segment Drawing**: Connects consecutive points with line segments
   - Creates smooth lane appearance through connected segments
3. **Thickness Control**: Uses specified line width for visibility

**OpenCV Line Drawing**:
- `cv2.line(img, pt1, pt2, color, thickness)`
- Draws anti-aliased lines for smooth appearance
- Modifies image in-place for efficiency

#### Display Handling

```python
if show:
    cv2.imshow("view", img)
    cv2.waitKey(0)
```

**Interactive Display**:
- **Window Creation**: Opens OpenCV window titled "view"
- **Blocking Wait**: `cv2.waitKey(0)` waits for any key press
- **Manual Control**: User controls when to proceed

**Use Cases**:
- **Debugging**: Step-by-step result inspection
- **Demonstration**: Interactive result viewing
- **Analysis**: Detailed examination of specific cases

#### File Output

```python
if out_file:
    if not osp.exists(osp.dirname(out_file)):
        os.makedirs(osp.dirname(out_file))
    cv2.imwrite(out_file, img)
```

**File Saving Process**:
1. **Directory Creation**: Ensures output directory exists
   - Uses `os.makedirs()` for recursive directory creation
   - Handles nested directory structures gracefully
2. **Image Writing**: Saves image to specified path
   - Preserves original image quality and format
   - Automatically determines format from file extension

**Supported Formats**: Any format supported by OpenCV (PNG, JPG, BMP, etc.)

## Integration with SRLane System

### Evaluation Pipeline

```python
# In evaluation/testing code
def visualize_results(model, dataloader, output_dir):
    for batch_idx, (images, targets) in enumerate(dataloader):
        predictions = model(images)
        
        for img, pred, target in zip(images, predictions, targets):
            # Convert to visualization format
            img_np = img.permute(1, 2, 0).cpu().numpy()
            pred_lanes = convert_predictions_to_lanes(pred)
            
            # Visualize
            out_path = osp.join(output_dir, f"result_{batch_idx}.jpg")
            imshow_lanes(img_np, pred_lanes, show=False, out_file=out_path)
```

### Training Monitoring

```python
# During training for debugging
def debug_batch(images, predictions, epoch, batch_idx):
    if batch_idx % 100 == 0:  # Visualize every 100 batches
        img = images[0].permute(1, 2, 0).cpu().numpy()
        lanes = extract_lanes_from_prediction(predictions[0])
        
        debug_path = f"debug/epoch_{epoch}_batch_{batch_idx}.jpg"
        imshow_lanes(img, lanes, show=False, out_file=debug_path)
```

### Dataset Analysis

```python
# Analyze dataset annotations
def analyze_dataset_lanes(dataset):
    for idx, (img, annotation) in enumerate(dataset):
        lanes = annotation['lanes']
        
        # Visualize ground truth
        imshow_lanes(img, lanes, show=True)  # Interactive viewing
        
        if idx >= 10:  # Limit analysis
            break
```

## Usage Examples

### Basic Visualization

```python
import cv2
from srlane.utils.visualization import imshow_lanes

# Load image
img = cv2.imread('test_image.jpg')

# Define lanes (example data)
lanes = [
    [(100, 400), (120, 300), (140, 200), (160, 100)],  # Lane 1
    [(200, 400), (220, 300), (240, 200), (260, 100)],  # Lane 2
    [(300, 400), (320, 300), (340, 200), (360, 100)],  # Lane 3
]

# Visualize with display
imshow_lanes(img, lanes, show=True)

# Save to file
imshow_lanes(img, lanes, out_file='output_with_lanes.jpg')
```

### Batch Processing

```python
def process_image_directory(input_dir, output_dir, model):
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png')):
            # Load and process image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            
            # Get predictions (example)
            lanes = model.predict(img)
            
            # Visualize and save
            out_path = os.path.join(output_dir, f"result_{filename}")
            imshow_lanes(img, lanes, out_file=out_path, width=6)
```

### Comparison Visualization

```python
def compare_predictions(img, ground_truth_lanes, predicted_lanes):
    # Create side-by-side comparison
    h, w = img.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=img.dtype)
    
    # Left side: ground truth
    gt_img = img.copy()
    imshow_lanes(gt_img, ground_truth_lanes, width=4)
    comparison[:, :w] = gt_img
    
    # Right side: predictions
    pred_img = img.copy()
    imshow_lanes(pred_img, predicted_lanes, width=4)
    comparison[:, w:] = pred_img
    
    # Save comparison
    cv2.imwrite('comparison.jpg', comparison)
```

### Video Processing

```python
def visualize_video(video_path, model, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get lane predictions
        lanes = model.predict(frame)
        
        # Visualize on frame
        imshow_lanes(frame, lanes, width=3)
        
        # Write to output video
        out.write(frame)
    
    cap.release()
    out.release()
```

## Advanced Customization

### Custom Color Schemes

```python
# Define custom colors for specific applications
CUSTOM_COLORS = [
    (0, 255, 0),    # Green for ego lane
    (255, 0, 0),    # Red for adjacent lanes
    (0, 0, 255),    # Blue for distant lanes
]

def imshow_lanes_custom(img, lanes, lane_types=None):
    for idx, lane in enumerate(lanes):
        if lane_types and idx < len(lane_types):
            color_idx = lane_types[idx]
        else:
            color_idx = idx % len(CUSTOM_COLORS)
        
        color = CUSTOM_COLORS[color_idx]
        # ... render with custom color
```

### Confidence-Based Visualization

```python
def imshow_lanes_with_confidence(img, lanes, confidences, width=4):
    for idx, (lane, conf) in enumerate(zip(lanes, confidences)):
        # Adjust line width based on confidence
        line_width = int(width * conf)
        
        # Adjust color intensity based on confidence  
        color = tuple(int(c * conf) for c in COLORS[idx])
        
        # Render with adjusted properties
        for i in range(1, len(lane)):
            cv2.line(img, lane[i-1], lane[i], color, thickness=line_width)
```

## Performance Considerations

- **In-Place Modification**: Image modified directly to avoid memory copying
- **Integer Coordinates**: Conversion to int avoids floating-point rendering issues
- **Batch Processing**: Suitable for processing large numbers of images
- **Memory Efficiency**: No unnecessary image copies or temporary arrays

## Error Handling and Robustness

- **Invalid Coordinates**: Gracefully filters out negative or zero coordinates
- **Empty Lanes**: Handles lanes with no valid points
- **Directory Creation**: Automatically creates output directories
- **File Format**: Flexible output format support through OpenCV

This visualization module provides essential tools for understanding, debugging, and presenting lane detection results in the SRLane system.