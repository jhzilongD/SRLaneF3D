# SRLane CULane Dataset Implementation

## File Overview
This file implements the `CULane` dataset class, which handles the CULane dataset format for lane detection. CULane is a large-scale challenging dataset for lane detection with diverse scenarios including normal, crowded, highlight, shadow, no-line, arrow, curve, cross, and night conditions.

## Code Structure

### Dataset Constants
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

#### Purpose:
- **`LIST_FILE`**: Maps dataset splits to their corresponding file lists
- **`CATEGORYS`**: Defines different test scenarios for comprehensive evaluation

### Class Definition
```python
@DATASETS.register_module
class CULane(BaseDataset):
```
Inherits from `BaseDataset` and is registered in the dataset registry for automatic instantiation.

### Constructor
```python
def __init__(self, data_root, split, processes=None, cfg=None):
```

#### Initialization Process:
1. **Parent Initialization**: Calls `BaseDataset.__init__()`
2. **Path Setup**: Constructs list file path using `LIST_FILE` mapping
3. **Annotation Loading**: Calls `load_annotations()` to populate `self.data_infos`
4. **Sample Points**: Sets `h_samples` for evaluation (y-coordinates from 270 to 590 pixels)

#### Key Attributes:
- **`list_path`**: Path to the dataset split file
- **`split`**: Current dataset split ("train", "val", "test")
- **`h_samples`**: Normalized y-coordinates for evaluation sampling

### Annotation Loading
```python
def load_annotations(self, diff_thr=15):
```

#### Purpose:
Loads and processes all annotations for the dataset split, with caching for performance optimization.

#### Caching Mechanism:
- **Cache Location**: `.cache/culane_{split}.pkl`
- **Cache Benefits**: Significantly speeds up subsequent dataset initializations
- **Cache Invalidation**: Manual deletion required when dataset changes

#### Processing Pipeline:
1. **File Parsing**: Reads the dataset list file line by line
2. **Path Construction**: Builds absolute paths for images and annotations
3. **Duplicate Detection**: For training split, filters near-duplicate images using pixel difference threshold
4. **Annotation Processing**: 
   - Loads lane coordinates from `.lines.txt` files
   - Filters invalid points (negative coordinates)
   - Removes duplicate points within lanes
   - Filters lanes with insufficient points (< 3 points)
   - Sorts lane points by y-coordinate (bottom to top)

#### Data Structure:
Each entry in `self.data_infos` contains:
```python
{
    "img_name": "relative/path/to/image.jpg",
    "img_path": "/absolute/path/to/image.jpg", 
    "mask_path": "/absolute/path/to/mask.png",  # Optional
    "lane_exist": np.array([0,1,1,0]),  # Lane existence flags
    "lanes": [[(x1,y1), (x2,y2), ...], ...]  # Lane coordinates
}
```

#### Duplicate Filtering Logic:
- **Threshold**: `diff_thr=15` (average pixel difference)
- **Calculation**: Mean absolute difference across all pixels and channels
- **Purpose**: Removes consecutive similar frames to improve training diversity

### Prediction String Generation
```python
def get_prediction_string(self, pred):
```

#### Purpose:
Converts model predictions to CULane evaluation format string representation.

#### Input Format:
- **`pred`**: List of lane objects with callable interface `lane(ys)` returning x-coordinates

#### Processing Steps:
1. **Y-Sampling**: Uses `self.h_samples` for consistent y-coordinate sampling
2. **Coordinate Conversion**: Calls `lane(ys)` to get x-coordinates at sample points
3. **Validity Filtering**: Removes points outside image boundaries (`xs >= 0` and `xs < 1`)
4. **Denormalization**: Converts normalized coordinates to pixel coordinates
5. **Ordering**: Reverses coordinates (bottom-to-top to top-to-bottom)
6. **String Formatting**: Creates space-separated coordinate pairs with 5 decimal precision

#### Output Format:
```
x1.xxxxx y1.xxxxx x2.xxxxx y2.xxxxx ...
x1.xxxxx y1.xxxxx x2.xxxxx y2.xxxxx ...
```

### Evaluation Method
```python
def evaluate(self, predictions, output_basedir):
```

#### Purpose:
Evaluates model predictions against ground truth using the official CULane metric.

#### Evaluation Pipeline:
1. **Output Generation**: 
   - Creates directory structure matching dataset layout
   - Writes prediction strings to `.lines.txt` files
   - Maintains original filename conventions

2. **Category Evaluation** (Test split only):
   - Evaluates each scenario category separately
   - Provides detailed performance breakdown per driving condition

3. **Overall Evaluation**:
   - Computes metrics across entire dataset split
   - Uses IoU threshold of 0.5 for lane matching
   - Returns F1 score as primary metric

#### Evaluation Metrics:
- **IoU Threshold**: 0.5 (standard for CULane)
- **Primary Metric**: F1 Score
- **Additional Metrics**: Precision, Recall, True Positives, False Positives, False Negatives

### Data Flow and Tensor Shapes

#### Annotation Loading Flow:
```
List File → Parse Lines → Extract Paths → Load Lane Coordinates → Filter & Sort → Cache
```

#### Prediction Evaluation Flow:
```
Model Predictions → Coordinate Sampling → Validation → Denormalization → String Format → File Output → Metric Calculation
```

#### Lane Coordinate Format:
- **Storage**: List of `(x, y)` tuples per lane
- **Coordinate System**: Image pixel coordinates
- **Sorting**: Y-coordinates sorted in descending order (bottom to top)
- **Filtering**: Removes invalid coordinates and duplicate points

## Configuration Dependencies

### Required Parameters:
- **`ori_img_h`**: Original image height (typically 590 for CULane)
- **`ori_img_w`**: Original image width (typically 1640 for CULane)

### Dataset Structure Requirements:
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
│   ├── *.jpg (images)
│   └── *.lines.txt (annotations)
└── laneseg_label_w16/ (optional masks)
```

## Usage Examples

### Dataset Creation:
```python
dataset = CULane(
    data_root="/path/to/culane",
    split="train",
    processes=augmentation_pipeline,
    cfg=config
)
```

### Evaluation Usage:
```python
f1_score = dataset.evaluate(
    predictions=model_predictions,
    output_basedir="/path/to/output"
)
```

### Prediction Format:
```python
# Each prediction should be a callable lane object
lane_prediction = lambda ys: np.array([...])  # Returns x-coords for given y-coords
predictions = [lane_prediction1, lane_prediction2, ...]
```

## Integration with SRLane System

### Training Integration:
- Inherits data loading and processing from `BaseDataset`
- Provides lane annotations in format expected by SRLane models
- Supports data augmentation through processing pipeline

### Evaluation Integration:
- Implements official CULane evaluation protocol
- Provides category-wise performance analysis
- Integrates with SRLane's validation pipeline

### Model Compatibility:
- Expects model predictions as callable lane objects
- Handles coordinate normalization and denormalization
- Supports various lane representation formats through prediction interface

## Performance Considerations

### Caching Strategy:
- Annotation loading cached to disk for faster startup
- Cache invalidation requires manual deletion
- Significant speedup for large datasets

### Memory Optimization:
- Lazy loading of images (loaded only when accessed)
- Efficient coordinate storage using tuples
- Minimal memory footprint for annotation metadata

### Evaluation Efficiency:
- Parallel processing in metric calculation
- Optimized coordinate conversion routines
- Batch processing of prediction files

This implementation provides a robust and efficient interface to the CULane dataset, handling the complexities of the dataset format while providing clean integration with the SRLane training and evaluation pipeline.