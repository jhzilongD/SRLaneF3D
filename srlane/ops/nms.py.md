# SRLane Non-Maximum Suppression (NMS) Implementation

## File Overview

This module provides the Python interface for Non-Maximum Suppression (NMS) operations in the SRLane lane detection system. NMS is a critical post-processing step that filters overlapping detections to remove redundant lane proposals, ensuring only the most confident and non-overlapping lane detections are retained.

## Copyright and Attribution

```python
# Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
# All rights reserved.
```

**Source**: Based on work from Durham University, indicating this is likely adapted from academic research or open-source computer vision implementations.

## Module Structure

### Imports

```python
from . import nms_impl
```

**Purpose**: Imports the actual CUDA implementation module `nms_impl` which contains the compiled CUDA kernels for NMS operations.

**Technical Details**:
- `nms_impl` is typically a compiled C++/CUDA extension module
- Built during package installation via `setup.py`
- Provides high-performance GPU-accelerated NMS operations

### Core Function

```python
def nms(boxes, scores, overlap, top_k):
    return nms_impl.nms_forward(boxes, scores, overlap, top_k)
```

**Function**: `nms`

**Purpose**: Performs Non-Maximum Suppression on bounding boxes and their associated confidence scores.

**Parameters**:
- `boxes` (torch.Tensor): Bounding boxes tensor with shape `[N, 4]` where N is number of boxes
  - Each row contains `[x1, y1, x2, y2]` coordinates
  - Coordinates are typically in pixel space
- `scores` (torch.Tensor): Confidence scores tensor with shape `[N]` corresponding to each box
  - Higher scores indicate more confident detections
- `overlap` (float): IoU (Intersection over Union) threshold for suppression
  - Boxes with IoU > overlap with higher-scored boxes are suppressed
  - Typical values range from 0.3 to 0.7
- `top_k` (int): Maximum number of detections to keep after NMS
  - Limits output to most confident detections

**Returns**:
- Indices of boxes that survive NMS filtering
- Typically returns tensor of shape `[M]` where M ≤ top_k

**Algorithm Flow**:
1. Sort boxes by confidence scores (descending)
2. For each box in sorted order:
   - Keep if not suppressed by higher-scored box
   - Suppress lower-scored boxes with IoU > threshold
3. Return indices of kept boxes

## Technical Implementation

### CUDA Acceleration

The function delegates to `nms_impl.nms_forward`, which implements:
- **Parallel Processing**: CUDA kernels process multiple box comparisons simultaneously
- **Memory Optimization**: Efficient GPU memory access patterns
- **Numerical Stability**: Robust IoU calculations avoiding floating-point issues

### Tensor Shape Transformations

**Input Shapes**:
```
boxes: [N, 4]    # N bounding boxes with [x1, y1, x2, y2]
scores: [N]      # N confidence scores
```

**Output Shape**:
```
indices: [M]     # M ≤ min(top_k, N) surviving box indices
```

## Role in SRLane System

### Detection Pipeline Integration

1. **Two-Stage Detection**: Used in both RPN and refinement stages
   - RPN stage: Filters initial lane proposals
   - Refinement stage: Final filtering of refined detections

2. **Multi-Scale Processing**: Applied at different feature pyramid levels
   - Each scale produces its own set of detections
   - NMS applied per scale before combining results

3. **Training vs Inference**:
   - **Training**: Less aggressive filtering to maintain gradient flow
   - **Inference**: More aggressive filtering for clean final results

### Performance Considerations

- **GPU Memory**: Efficient for large numbers of detections
- **Batch Processing**: Can handle multiple images in parallel
- **Real-time Inference**: Critical for meeting speed requirements

## Usage Examples

### Basic NMS Usage
```python
from srlane.ops import nms
import torch

# Example detection results
boxes = torch.tensor([[10, 10, 50, 50],
                     [15, 15, 55, 55],
                     [100, 100, 150, 150]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.95], dtype=torch.float32)

# Apply NMS
keep_indices = nms(boxes, scores, overlap=0.5, top_k=100)
# keep_indices might be [2, 0] (highest scoring non-overlapping boxes)

# Filter results
final_boxes = boxes[keep_indices]
final_scores = scores[keep_indices]
```

### Integration in Detection Head
```python
class DetectionHead(nn.Module):
    def forward(self, features):
        # Generate predictions
        box_preds = self.box_predictor(features)
        score_preds = self.score_predictor(features)
        
        # Apply NMS during inference
        if not self.training:
            keep_indices = nms(box_preds, score_preds, 
                              overlap=0.5, top_k=1000)
            box_preds = box_preds[keep_indices]
            score_preds = score_preds[keep_indices]
        
        return box_preds, score_preds
```

## Dependencies

- **PyTorch**: Core tensor operations and CUDA support
- **CUDA Toolkit**: Required for compilation of CUDA kernels
- **C++ Compiler**: Needed for building the extension module

## Performance Characteristics

- **Time Complexity**: O(N²) worst case, but CUDA parallelization makes it practical
- **Memory Usage**: Linear in number of input boxes
- **GPU Utilization**: High parallel efficiency on modern GPUs
- **Typical Performance**: Processes thousands of detections in milliseconds

## Error Handling

The CUDA implementation typically handles:
- Empty input tensors (returns empty indices)
- Single box inputs (returns index 0)
- Invalid coordinates (filters them out)
- GPU memory constraints (may fall back to CPU if needed)