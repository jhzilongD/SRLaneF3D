# SRLane Operations Module Initialization

## File Overview

This is the initialization file for the `srlane.ops` module, which contains custom CUDA operations for non-maximum suppression (NMS) used in the SRLane lane detection system. The module serves as the interface to optimized CUDA implementations for efficient post-processing operations.

## Module Structure

### Imports and Exports

```python
from .nms import nms

__all__ = ['nms']
```

**Purpose**: Exposes the `nms` (Non-Maximum Suppression) function as the primary public interface of the ops module.

**Data Flow**: 
- Imports the `nms` function from the local `nms.py` module
- Makes it available for external imports via `from srlane.ops import nms`

## Role in SRLane System

The ops module provides critical performance-optimized operations for the SRLane detection pipeline:

1. **Post-processing Operations**: Houses CUDA-accelerated implementations of computationally intensive operations
2. **Performance Critical Path**: These operations are called frequently during inference and need to be highly optimized
3. **GPU Acceleration**: Leverages CUDA implementations for parallel processing on GPU hardware

## Usage Examples

```python
# Import NMS operation
from srlane.ops import nms

# Use in detection pipeline
filtered_boxes = nms(boxes, scores, overlap_threshold, top_k)
```

## Technical Implementation

The module follows PyTorch's custom operation pattern:
- Provides Python interface through this init file
- Actual implementation delegated to `nms.py` which interfaces with CUDA kernels
- Built as part of the setup process using `setup.py` for CUDA compilation

## Dependencies

- Requires CUDA-compatible PyTorch installation
- CUDA kernels must be compiled during package installation
- Used by detection heads in the SRLane model for proposal filtering

## Integration Points

This module integrates with:
- `srlane.models.heads`: Detection heads use NMS for filtering proposals
- `srlane.models.nets.detector`: Main detector orchestrates NMS in post-processing
- Training and inference pipelines: Critical for both training and evaluation phases