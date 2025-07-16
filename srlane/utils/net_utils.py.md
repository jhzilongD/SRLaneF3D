# SRLane 网络工具

## 文件概述

该模块为 SRLane 系统中的神经网络模型管理提供基本工具。它处理模型检查点保存和加载操作，包括正确处理多GPU训练场景和状态字典键管理。这些工具对于训练持久性、模型部署和实验可重现性至关重要。

## 导入和依赖

```python
import os
import torch
import torch.nn.functional
```

**依赖项**:
- **os**: 用于路径管理和目录创建的文件系统操作
- **torch**: 用于张量操作和模型状态的 PyTorch 核心功能
- **torch.nn.functional**: （已导入但在当前实现中未使用）

## 核心函数

### 模型保存

```python
def save_model(net, recorder):
    model_dir = os.path.join(recorder.work_dir, "ckpt")
    os.system(f"mkdir -p {model_dir}")
    epoch = recorder.epoch
    ckpt_name = epoch
    torch.save(
        {
            "net": net.state_dict(),
        },
        os.path.join(model_dir, f"{ckpt_name}.pth"))
```

**Function**: `save_model`

**Purpose**: Saves a PyTorch model's state dictionary to a checkpoint file with organized directory structure.

**Parameters**:
- `net` (torch.nn.Module): The neural network model to save
  - Can be a single model or wrapped in DataParallel/DistributedDataParallel
- `recorder` (Recorder): Recording object containing training state and work directory
  - Provides `work_dir` for output location
  - Provides `epoch` for checkpoint naming

**Implementation Details**:

1. **Directory Setup**:
   ```python
   model_dir = os.path.join(recorder.work_dir, "ckpt")
   os.system(f"mkdir -p {model_dir}")
   ```
   - Creates checkpoint directory within work directory
   - Uses `mkdir -p` to create parent directories if needed
   - Follows Unix convention for directory creation

2. **Checkpoint Naming**:
   ```python
   epoch = recorder.epoch
   ckpt_name = epoch
   ```
   - Names checkpoint files based on current epoch number
   - Enables easy identification of training progress
   - Facilitates chronological ordering of checkpoints

3. **State Dictionary Packaging**:
   ```python
   torch.save(
       {
           "net": net.state_dict(),
       },
       os.path.join(model_dir, f"{ckpt_name}.pth"))
   ```
   - Packages model state in dictionary format
   - Uses "net" key for model parameters
   - Enables future extension with optimizer state, scheduler state, etc.
   - Saves in PyTorch's standard .pth format

**File Structure Output**:
```
work_dir/
└── ckpt/
    ├── 1.pth      # Epoch 1 checkpoint
    ├── 2.pth      # Epoch 2 checkpoint
    └── ...
```

### Model Loading

```python
def load_network(net, model_dir, strict=False):
    weights = torch.load(model_dir)["net"]
    new_weights = {}
    for k, v in weights.items():
        new_k = k.replace("module.", '') if "module" in k else k
        new_weights[new_k] = v
    net.load_state_dict(new_weights, strict=strict)
```

**Function**: `load_network`

**Purpose**: Loads a saved model checkpoint into a neural network, handling multi-GPU training artifacts and key name mismatches.

**Parameters**:
- `net` (torch.nn.Module): Target network to load weights into
- `model_dir` (str): Path to the checkpoint file (.pth)
- `strict` (bool): Whether to strictly enforce key matching
  - Default: False (allows partial loading and key mismatches)
  - True: Requires exact key matching between checkpoint and model

**Implementation Details**:

1. **Checkpoint Loading**:
   ```python
   weights = torch.load(model_dir)["net"]
   ```
   - Loads checkpoint dictionary from file
   - Extracts model weights using "net" key
   - Compatible with save_model format

2. **Key Name Processing**:
   ```python
   new_weights = {}
   for k, v in weights.items():
       new_k = k.replace("module.", '') if "module" in k else k
       new_weights[new_k] = v
   ```
   - **Purpose**: Handles DataParallel/DistributedDataParallel artifacts
   - **Problem**: Multi-GPU training wraps models, adding "module." prefix to parameter names
   - **Solution**: Strips "module." prefix when present
   - **Compatibility**: Works with both single-GPU and multi-GPU saved models

3. **State Dictionary Loading**:
   ```python
   net.load_state_dict(new_weights, strict=strict)
   ```
   - Loads processed weights into target network
   - `strict=False` allows flexibility for partial loading
   - Enables transfer learning and model evolution scenarios

## Multi-GPU Training Considerations

### DataParallel Wrapper Effects

When using `torch.nn.DataParallel`, PyTorch wraps the model:
```python
# Original model
class MyModel(nn.Module): ...

# DataParallel wrapped
model = nn.DataParallel(MyModel())

# Parameter names become:
# "layer1.weight" → "module.layer1.weight"
# "layer1.bias"   → "module.layer1.bias"
```

### Loading Flexibility

The `load_network` function handles these scenarios:
1. **Single → Single**: Direct loading (no key changes)
2. **Multi → Single**: Strips "module." prefix
3. **Single → Multi**: Target model adds "module." automatically
4. **Multi → Multi**: Direct loading (keys match)

## Integration with SRLane System

### Training Pipeline Integration

```python
# In training loop
def train_epoch(model, recorder):
    # ... training code ...
    
    # Save checkpoint at epoch end
    save_model(model, recorder)
    recorder.logger.info(f"Checkpoint saved: epoch {recorder.epoch}")

# Resume training
def resume_training(model, checkpoint_path):
    load_network(model, checkpoint_path, strict=False)
    print("Training resumed from checkpoint")
```

### Recorder Integration

The functions integrate with the Recorder class:
- **Work Directory**: Automatic checkpoint organization
- **Epoch Tracking**: Checkpoint naming based on training progress
- **Logging**: Integration with logging system for checkpoint operations

### Configuration System

```python
# In config files
checkpoint_config = dict(
    save_interval=10,      # Save every 10 epochs
    max_keep_ckpts=5,      # Keep only last 5 checkpoints
    resume_from=None,      # Path to checkpoint for resuming
)
```

## Usage Examples

### Basic Model Saving and Loading

```python
import torch
from srlane.utils.net_utils import save_model, load_network
from srlane.utils.recorder import Recorder

# Setup
model = SRLaneDetector()
cfg = Config()
recorder = Recorder(cfg)

# Training loop
for epoch in range(100):
    # ... training code ...
    recorder.epoch = epoch
    
    # Save checkpoint
    save_model(model, recorder)

# Later: load checkpoint
load_network(model, "work_dirs/20240716_103045/ckpt/50.pth")
```

### Multi-GPU Training Scenario

```python
# Multi-GPU training setup
model = SRLaneDetector()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Training and saving (handles DataParallel automatically)
save_model(model, recorder)

# Loading into single-GPU model (strips "module." prefix)
single_gpu_model = SRLaneDetector()
load_network(single_gpu_model, checkpoint_path, strict=False)
```

### Flexible Loading with Error Handling

```python
def safe_load_checkpoint(model, checkpoint_path):
    try:
        # Try strict loading first
        load_network(model, checkpoint_path, strict=True)
        print("Checkpoint loaded successfully (strict)")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        try:
            # Fall back to non-strict loading
            load_network(model, checkpoint_path, strict=False)
            print("Checkpoint loaded successfully (non-strict)")
        except Exception as e2:
            print(f"Loading failed completely: {e2}")
            raise
```

### Transfer Learning

```python
# Load pretrained backbone
pretrained_model = SRLaneDetector()
load_network(pretrained_model, "pretrained_backbone.pth", strict=False)

# Fine-tune specific layers
for name, param in pretrained_model.named_parameters():
    if "head" not in name:  # Freeze backbone
        param.requires_grad = False
```

## Advanced Features

### Checkpoint Management

```python
def cleanup_old_checkpoints(work_dir, keep_last_n=5):
    """Keep only the last N checkpoints to save disk space."""
    ckpt_dir = os.path.join(work_dir, "ckpt")
    if not os.path.exists(ckpt_dir):
        return
    
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    ckpt_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort by epoch
    
    for old_ckpt in ckpt_files[:-keep_last_n]:
        os.remove(os.path.join(ckpt_dir, old_ckpt))
```

### Extended Checkpoint Information

```python
def save_model_extended(net, recorder, optimizer=None, scheduler=None):
    """Enhanced version with optimizer and scheduler state."""
    model_dir = os.path.join(recorder.work_dir, "ckpt")
    os.system(f"mkdir -p {model_dir}")
    
    checkpoint = {
        "net": net.state_dict(),
        "epoch": recorder.epoch,
        "step": recorder.step,
    }
    
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    
    torch.save(checkpoint, os.path.join(model_dir, f"{recorder.epoch}.pth"))
```

## Error Handling and Robustness

### Common Issues and Solutions

1. **Missing Directories**: Uses `mkdir -p` for robust directory creation
2. **Key Mismatches**: Non-strict loading allows partial state loading
3. **Memory Issues**: Loads checkpoints to CPU first, then transfers to GPU
4. **File Permissions**: Handles read-only filesystems gracefully

### Best Practices

- **Regular Checkpointing**: Save frequently to prevent loss of progress
- **Validation Before Save**: Ensure model is in good state before saving
- **Atomic Operations**: Use temporary files and atomic moves for crash safety
- **Version Compatibility**: Include PyTorch version info in checkpoints

This network utilities module provides robust, flexible model management capabilities essential for productive deep learning experimentation and deployment in the SRLane system.