# SRLane Tensor Transformation Utilities

## File Overview
This file implements tensor conversion utilities for the SRLane data processing pipeline. It provides the final transformation step that converts processed numpy arrays and other Python data types into PyTorch tensors with proper formatting for training and inference.

## Code Structure

### Utility Function
```python
def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """
```

#### Purpose:
Provides a unified interface for converting various Python data types to PyTorch tensors.

#### Supported Type Conversions:
- **`torch.Tensor`**: Returns input unchanged (pass-through)
- **`numpy.ndarray`**: Converts using `torch.from_numpy()` (memory-efficient)
- **`int`**: Wraps in `torch.LongTensor([data])`
- **`float`**: Wraps in `torch.FloatTensor([data])`
- **Unsupported**: Raises `TypeError` with descriptive message

#### Implementation Details:
```python
if isinstance(data, torch.Tensor):
    return data
elif isinstance(data, np.ndarray):
    return torch.from_numpy(data)
elif isinstance(data, int):
    return torch.LongTensor([data])
elif isinstance(data, float):
    return torch.FloatTensor([data])
else:
    raise TypeError(f"type {type(data)} cannot be converted to tensor.")
```

#### Memory Efficiency:
- **NumPy Arrays**: Uses `torch.from_numpy()` which shares memory when possible
- **Scalar Values**: Creates minimal single-element tensors
- **Existing Tensors**: No-op for already converted data

### ToTensor Transform Class
```python
@PROCESS.register_module
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
```

Registered in the PROCESS registry for automatic instantiation from configuration files.

#### Constructor
```python
def __init__(self, keys=["img", "mask"], cfg=None):
```

##### Parameters:
- **`keys`** (Sequence[str]): List of dictionary keys to convert to tensors
- **`cfg`** (object, optional): Configuration object (not used in current implementation)

##### Default Behavior:
- Converts `"img"` and `"mask"` keys by default
- Commonly overridden to include ground truth keys like `"gt_lane"`, `"gt_angle"`, `"gt_seg"`

#### Main Transform Method
```python
def __call__(self, sample):
```

##### Purpose:
Converts specified keys in a sample dictionary from numpy arrays to PyTorch tensors with proper formatting.

##### Processing Pipeline:
1. **Image Dimension Expansion**: Ensures images have 3 dimensions
2. **Selective Conversion**: Converts only specified keys to tensors
3. **Container Preservation**: Preserves lists and dictionaries without conversion
4. **Image Reordering**: Transposes image tensors from HWC to CHW format

##### Implementation Details:
```python
def __call__(self, sample):
    data = {}
    
    # Ensure image has channel dimension
    if len(sample["img"].shape) < 3:
        sample["img"] = np.expand_dims(sample["img"], -1)
    
    # Convert specified keys
    for key in self.keys:
        if isinstance(sample[key], list) or isinstance(sample[key], dict):
            data[key] = sample[key]  # Preserve containers
            continue
        data[key] = to_tensor(sample[key])  # Convert to tensor
    
    # Reorder image dimensions: HWC -> CHW
    data["img"] = data["img"].permute(2, 0, 1)
    
    return data
```

### Tensor Shape Transformations

#### Image Tensor Transformation:
```
Input:  np.ndarray (H, W, C) or (H, W) 
        ↓
Expand: np.ndarray (H, W, C) if needed
        ↓  
Convert: torch.Tensor (H, W, C)
        ↓
Permute: torch.Tensor (C, H, W)
```

#### Ground Truth Tensor Transformation:
```
NumPy Arrays → PyTorch Tensors (same shape)
- gt_lane: (max_lanes, 4+num_points) 
- gt_angle: List[(H//stride, W//stride) for stride in [8,16,32]]
- gt_seg: List[(H//stride, W//stride) for stride in [8,16,32]]
```

#### Data Type Preservation:
- **Float Arrays**: Converted to `torch.FloatTensor`
- **Integer Arrays**: Converted to appropriate integer tensor type
- **Boolean Arrays**: Converted to `torch.BoolTensor`

## Usage Patterns

### Basic Configuration:
```python
# Minimal configuration for image-only processing
transform = ToTensor(keys=["img"])

# Full configuration for training
transform = ToTensor(keys=["img", "gt_lane", "gt_angle", "gt_seg"])
```

### Configuration File Usage:
```python
# In dataset configuration
processes = [
    {
        "type": "GenerateLaneLine",
        "transforms": [...]
    },
    {
        "type": "ToTensor",
        "keys": ["img", "gt_lane", "gt_angle", "gt_seg", "meta"]
    }
]
```

### Processing Sample:
```python
# Input sample (numpy arrays)
input_sample = {
    "img": np.ndarray,      # Shape: (320, 800, 3)
    "gt_lane": np.ndarray,  # Shape: (4, 76)
    "gt_angle": [np.ndarray, ...],  # Multi-scale arrays
    "gt_seg": [np.ndarray, ...],    # Multi-scale arrays
    "metadata": {...}       # Dictionary (preserved)
}

# Apply transformation
transform = ToTensor(keys=["img", "gt_lane", "gt_angle", "gt_seg"])
output_sample = transform(input_sample)

# Output sample (PyTorch tensors)
output_sample = {
    "img": torch.Tensor,      # Shape: (3, 320, 800)
    "gt_lane": torch.Tensor,  # Shape: (4, 76)
    "gt_angle": [torch.Tensor, ...],  # Multi-scale tensors
    "gt_seg": [torch.Tensor, ...],    # Multi-scale tensors
    "metadata": {...}         # Dictionary (unchanged)
}
```

## Integration with SRLane System

### Training Pipeline Integration:
The `ToTensor` transform is typically the final step in the processing pipeline:

```python
# Complete processing pipeline
processes = [
    GenerateLaneLine(transforms=augmentations),  # Data processing
    ToTensor(keys=["img", "gt_lane", "gt_angle", "gt_seg"])  # Tensor conversion
]
```

### DataLoader Compatibility:
- **Batch Collation**: Converted tensors can be batched by PyTorch DataLoader
- **GPU Transfer**: Tensors ready for `.to(device)` operations
- **Memory Layout**: CHW image format expected by CNN models

### Model Input Preparation:
```python
# After processing and batching
batch = {
    "img": torch.Tensor,      # Shape: (batch_size, 3, H, W)
    "gt_lane": torch.Tensor,  # Shape: (batch_size, max_lanes, 4+num_points)
    "gt_angle": [torch.Tensor, ...],  # Multi-scale ground truth
}

# Ready for model forward pass
output = model(batch["img"], batch)
```

## Configuration Dependencies

### Required Keys Configuration:
Typically configured based on model requirements:

```python
# For inference only
keys = ["img"]

# For training with full supervision
keys = ["img", "gt_lane", "gt_angle", "gt_seg"]

# For evaluation with metadata
keys = ["img", "gt_lane", "meta"]
```

### Integration with Dataset Classes:
```python
class CULane(BaseDataset):
    def __getitem__(self, idx):
        sample = self.processes(sample)  # Includes ToTensor
        return sample  # All specified keys now as tensors
```

## Error Handling

### Type Validation:
```python
# Automatic type checking in to_tensor()
try:
    tensor = to_tensor(data)
except TypeError as e:
    print(f"Cannot convert {type(data)} to tensor: {e}")
```

### Shape Validation:
The transform handles common shape issues:
- **Missing Channel Dimension**: Automatically adds channel dimension for grayscale images
- **Dimension Reordering**: Ensures correct channel order for CNN models

### Container Handling:
- **Lists**: Preserved without conversion (e.g., multi-scale ground truth)
- **Dictionaries**: Preserved without conversion (e.g., metadata)
- **Nested Structures**: Handles complex data structures appropriately

## Performance Considerations

### Memory Efficiency:
- **Zero-Copy**: `torch.from_numpy()` shares memory when possible
- **In-Place Operations**: Minimal additional memory allocation
- **Efficient Permutation**: Uses PyTorch's optimized tensor operations

### Computational Efficiency:
- **Selective Conversion**: Only converts specified keys
- **Type Checking**: Fast isinstance() checks for type dispatch
- **Batch Processing**: Efficient tensor operations for batch data

## Common Usage Examples

### Training Configuration:
```python
ToTensor(keys=["img", "gt_lane", "gt_angle", "gt_seg"])
```

### Inference Configuration:
```python
ToTensor(keys=["img"])
```

### Evaluation Configuration:
```python
ToTensor(keys=["img", "gt_lane", "meta"])
```

### Custom Processing:
```python
# Process only specific ground truth components
ToTensor(keys=["img", "gt_angle"])  # Skip lane coordinates
```

This tensor transformation module provides the essential final step in SRLane's data processing pipeline, ensuring that all data is properly formatted as PyTorch tensors with correct shapes and memory layout for efficient training and inference.