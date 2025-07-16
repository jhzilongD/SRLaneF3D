# SRLane 训练进度记录器

## 文件概述

该模块为 SRLane 系统提供全面的训练进度跟踪和记录功能。它实现了 `Recorder` 类和支持工具，处理实验管理、日志记录、指标跟踪和 TensorBoard 集成。记录器作为监控训练进度、管理实验产物和在整个训练流程中提供详细日志的中心枢纽。

## Imports and Dependencies

```python
import os
import datetime
import logging
import pathspec
import torch

from collections import deque, defaultdict
from torch.utils.tensorboard import SummaryWriter

from .logger import init_logger
```

**Dependencies**:
- **os**: File system operations and path management
- **datetime**: Timestamp generation and time calculations
- **logging**: Integration with Python's logging system
- **pathspec**: Git-style pattern matching for file operations
- **torch**: PyTorch tensor operations and utilities
- **collections**: Advanced data structures (deque, defaultdict)
- **torch.utils.tensorboard**: TensorBoard logging integration
- **logger**: Local logging infrastructure

## Supporting Classes

### SmoothedValue Class

```python
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
```

**Purpose**: Provides smoothed statistical tracking of training metrics using a sliding window approach.

**Parameters**:
- `window_size` (int): Size of the sliding window for recent value tracking
  - Default: 20 (tracks last 20 values)
  - Balances responsiveness with smoothing

**Data Structures**:
- `deque`: Fixed-size sliding window for recent values
- `total`: Cumulative sum of all values ever added
- `count`: Total number of values processed

#### Value Update

```python
def update(self, value):
    self.deque.append(value)
    self.count += 1
    self.total += value
```

**Function**: `update`

**Purpose**: Adds a new value to the tracking system, updating both windowed and global statistics.

**Parameters**:
- `value` (float): New metric value to track

**Operations**:
1. **Window Update**: Adds value to sliding window (automatically removes oldest if full)
2. **Global Tracking**: Updates cumulative statistics for global averages

#### Statistical Properties

```python
@property
def median(self):
    d = torch.tensor(list(self.deque))
    return d.median().item()

@property
def avg(self):
    d = torch.tensor(list(self.deque))
    return d.mean().item()

@property
def global_avg(self):
    return self.total / self.count
```

**Properties**:

1. **median**: Median of values in current window
   - Robust to outliers
   - Good for understanding typical recent performance

2. **avg**: Mean of values in current window
   - Standard moving average
   - Responsive to recent trends

3. **global_avg**: Average of all values since initialization
   - Stable long-term average
   - Less affected by recent fluctuations

**Tensor Operations**: Uses PyTorch tensors for efficient statistical computations.

## Core Recorder Class

### Initialization

```python
class Recorder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = self.get_work_dir()
        cfg.work_dir = self.work_dir
        self.log_path = os.path.join(self.work_dir, "log.txt")
        self.tb_logger = SummaryWriter(log_dir=self.work_dir)

        init_logger(self.log_path)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Config: \n" + cfg.text)

        self.save_cfg(cfg)
        #    self.cp_projects(self.work_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_status = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()
        self.max_iter = self.cfg.total_iter
        self.lr = 0.
```

**Purpose**: Initializes a comprehensive experiment recording system with directory management, logging setup, and metric tracking.

**Parameters**:
- `cfg` (Config): Configuration object containing experiment settings

**Initialization Process**:

1. **Directory Management**:
   ```python
   self.work_dir = self.get_work_dir()
   cfg.work_dir = self.work_dir
   ```
   - Creates unique timestamped work directory
   - Updates config with actual work directory path

2. **Logging Setup**:
   ```python
   self.log_path = os.path.join(self.work_dir, "log.txt")
   init_logger(self.log_path)
   self.logger = logging.getLogger(__name__)
   ```
   - Establishes file-based logging
   - Creates module-specific logger
   - Logs initial configuration

3. **TensorBoard Integration**:
   ```python
   self.tb_logger = SummaryWriter(log_dir=self.work_dir)
   ```
   - Sets up TensorBoard logging to work directory
   - Enables real-time metric visualization

4. **Metric Tracking**:
   ```python
   self.loss_status = defaultdict(SmoothedValue)
   self.batch_time = SmoothedValue()
   self.data_time = SmoothedValue()
   ```
   - Creates smoothed trackers for various metrics
   - Automatically creates new trackers for previously unseen loss types

### Work Directory Management

```python
def get_work_dir(self):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(self.cfg.work_dirs, now)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    return work_dir
```

**Function**: `get_work_dir`

**Purpose**: Creates a unique timestamped directory for experiment outputs.

**Directory Structure**:
```
work_dirs/
└── 20240716_103045/    # Timestamp: YYYYMMDD_HHMMSS
    ├── log.txt         # Training logs
    ├── config.py       # Saved configuration
    ├── ckpt/           # Model checkpoints
    └── events.out.*    # TensorBoard logs
```

**Timestamp Format**: `%Y%m%d_%H%M%S` ensures:
- Chronological ordering
- No directory name conflicts
- Human-readable experiment identification

### Configuration Management

```python
def save_cfg(self, cfg):
    cfg_path = os.path.join(self.work_dir, "config.py")
    with open(cfg_path, 'w') as cfg_file:
        cfg_file.write(cfg.text)
```

**Function**: `save_cfg`

**Purpose**: Saves the complete experiment configuration for reproducibility.

**Benefits**:
- **Reproducibility**: Exact configuration preserved
- **Debugging**: Easy comparison between experiments
- **Documentation**: Self-documenting experiment parameters

### Project Code Archival (Commented)

```python
def cp_projects(self, to_path):
    with open("./.gitignore", 'r') as fp:
        ign = fp.read()
    ign += "\n.git"
    spec = pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern, ign.splitlines())
    all_files = {
        os.path.join(root, name)
        for root, dirs, files in os.walk("./") for name in files
    }
    matches = spec.match_files(all_files)
    matches = set(matches)
    to_cp_files = all_files - matches
    for f in to_cp_files:
        dirs = os.path.join(to_path, "code", os.path.split(f[2:])[0])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        os.system("cp %s %s" % (f, os.path.join(to_path, "code", f[2:])))
```

**Function**: `cp_projects` (Currently commented out)

**Purpose**: Archives the entire codebase snapshot for complete experiment reproducibility.

**Implementation**:
1. **Git Integration**: Respects .gitignore patterns
2. **Selective Copying**: Excludes version control and temporary files
3. **Structure Preservation**: Maintains directory hierarchy
4. **Code Snapshot**: Complete source code at experiment time

### Loss Tracking

```python
def update_loss_status(self, loss_dict):
    for k, v in loss_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        self.loss_status[k].update(v.detach().mean().cpu())
```

**Function**: `update_loss_status`

**Purpose**: Updates smoothed loss tracking from model outputs.

**Parameters**:
- `loss_dict` (dict): Dictionary of loss components from model
  - Keys: Loss component names (e.g., 'total_loss', 'cls_loss', 'reg_loss')
  - Values: Loss tensors (may be multi-dimensional)

**Processing Steps**:
1. **Type Filtering**: Only processes PyTorch tensors
2. **Gradient Detachment**: Removes from computation graph
3. **Dimension Reduction**: Computes mean for multi-dimensional losses
4. **Device Transfer**: Moves to CPU for efficient storage
5. **Smoothed Tracking**: Updates corresponding SmoothedValue tracker

### Logging and Recording

```python
def record(self, prefix):
    self.logger.info(self)
    self.tb_logger.add_scalar(f"{prefix}/lr", self.lr, self.step)
    for k, v in self.loss_status.items():
        self.tb_logger.add_scalar(f"{prefix}/" + k, v.avg, self.step)
```

**Function**: `record`

**Purpose**: Logs current training status to both text logs and TensorBoard.

**Parameters**:
- `prefix` (str): Logging prefix (e.g., 'train', 'val', 'test')

**Logging Operations**:
1. **Text Logging**: Records formatted status string
2. **Learning Rate**: Logs current learning rate to TensorBoard
3. **Loss Components**: Logs smoothed averages of all tracked losses

**TensorBoard Organization**:
```
train/lr           # Learning rate
train/total_loss   # Total loss
train/cls_loss     # Classification loss
train/reg_loss     # Regression loss
val/total_loss     # Validation losses...
```

### File Logging

```python
def write(self, content):
    with open(self.log_path, "a+") as f:
        f.write(content)
        f.write('\n')
```

**Function**: `write`

**Purpose**: Appends arbitrary content to the experiment log file.

**Usage**: Direct logging of custom information beyond standard metrics.

### State Management

```python
def state_dict(self):
    scalar_dict = {}
    scalar_dict["step"] = self.step
    return scalar_dict

def load_state_dict(self, scalar_dict):
    self.step = scalar_dict["step"]
```

**Functions**: `state_dict` and `load_state_dict`

**Purpose**: Enables saving and restoring recorder state for training resumption.

**Current State**: Minimal implementation (only tracks step count)
**Extensible**: Can be expanded to include loss history, timing information, etc.

### Status String Formatting

```python
def __str__(self):
    loss_state = []
    for k, v in self.loss_status.items():
        loss_state.append(f"{k}: {v.avg:.4f}")
    loss_state = "  ".join(loss_state)

    recording_state = "  ".join([
        "epoch: {}", "step: {}", "lr: {:.6f}", "{}", "data: {:.4f}",
        "batch: {:.4f}", "eta: {}"
    ])
    eta_seconds = self.batch_time.global_avg * (self.max_iter - self.step)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    return recording_state.format(self.epoch, self.step, self.lr,
                                  loss_state, self.data_time.avg,
                                  self.batch_time.avg, eta_string)
```

**Function**: `__str__`

**Purpose**: Creates comprehensive human-readable status string for logging.

**Output Format**:
```
epoch: 1  step: 100  lr: 0.001000  total_loss: 2.3456  cls_loss: 1.2345  reg_loss: 1.1111  data: 0.0123  batch: 0.0456  eta: 2:34:56
```

**Components**:
- **Training Progress**: Current epoch and step
- **Learning Rate**: Current optimizer learning rate
- **Loss Breakdown**: All tracked loss components with 4 decimal precision
- **Timing**: Data loading and batch processing times
- **ETA**: Estimated time to completion based on current progress

**ETA Calculation**:
```python
eta_seconds = self.batch_time.global_avg * (self.max_iter - self.step)
eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
```
- Uses global average batch time for stable estimates
- Calculates remaining iterations × time per iteration
- Formats as human-readable duration (HH:MM:SS)

## Factory Function

```python
def build_recorder(cfg):
    return Recorder(cfg)
```

**Function**: `build_recorder`

**Purpose**: Factory function for creating Recorder instances with consistent interface.

**Benefits**:
- **Consistency**: Standard creation pattern
- **Extensibility**: Easy to modify creation logic
- **Configuration**: Single point for recorder configuration

## Integration with SRLane System

### Training Loop Integration

```python
def train_epoch(model, dataloader, optimizer, recorder):
    for batch_idx, (data, target) in enumerate(dataloader):
        start_time = time.time()
        
        # Data loading time
        recorder.data_time.update(time.time() - start_time)
        
        # Forward pass
        batch_start = time.time()
        loss_dict = model(data, target)
        
        # Update losses
        recorder.update_loss_status(loss_dict)
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # Batch processing time
        recorder.batch_time.update(time.time() - batch_start)
        recorder.step += 1
        
        # Periodic logging
        if batch_idx % 100 == 0:
            recorder.record('train')
```

### Experiment Management

```python
def setup_experiment(cfg):
    # Create recorder
    recorder = build_recorder(cfg)
    
    # Log experiment start
    recorder.logger.info(f"Starting experiment: {cfg.exp_name}")
    recorder.logger.info(f"Work directory: {recorder.work_dir}")
    
    return recorder
```

## Usage Examples

### Basic Training Setup

```python
from srlane.utils.recorder import build_recorder

# Setup experiment
cfg = load_config('configs/exp_srlane_culane.py')
recorder = build_recorder(cfg)

# Training loop
for epoch in range(cfg.epochs):
    recorder.epoch = epoch
    
    for batch_idx, batch in enumerate(dataloader):
        # ... training code ...
        
        # Update metrics
        recorder.update_loss_status(loss_dict)
        recorder.step += 1
        
        # Periodic logging
        if batch_idx % cfg.log_interval == 0:
            recorder.record('train')
```

### Custom Metric Tracking

```python
# Track custom metrics
recorder.loss_status['custom_metric'].update(custom_value)
recorder.loss_status['iou_score'].update(iou_value)

# These automatically appear in logs and TensorBoard
recorder.record('train')
```

### Timing Analysis

```python
import time

# Track data loading time
data_start = time.time()
batch = next(dataloader_iter)
recorder.data_time.update(time.time() - data_start)

# Track model forward time
forward_start = time.time()
outputs = model(batch)
forward_time = time.time() - forward_start
recorder.loss_status['forward_time'].update(forward_time)
```

## Advanced Features

### Multi-GPU Training Support

```python
# Aggregate losses across GPUs
def update_distributed_losses(recorder, loss_dict):
    for k, v in loss_dict.items():
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(v)
            v /= torch.distributed.get_world_size()
        recorder.update_loss_status({k: v})
```

### Validation Integration

```python
def validate(model, val_loader, recorder):
    model.eval()
    val_losses = defaultdict(list)
    
    with torch.no_grad():
        for batch in val_loader:
            loss_dict = model(batch)
            for k, v in loss_dict.items():
                val_losses[k].append(v.item())
    
    # Update recorder with validation results
    for k, v_list in val_losses.items():
        recorder.loss_status[f'val_{k}'].update(np.mean(v_list))
    
    recorder.record('val')
```

## Performance and Memory Considerations

- **Efficient Storage**: SmoothedValue uses fixed-size deques for memory efficiency
- **CPU Operations**: Loss values moved to CPU to avoid GPU memory accumulation
- **Batch Logging**: TensorBoard writes are batched for performance
- **File I/O**: Log files use buffered writes for efficiency

This comprehensive recording system provides essential infrastructure for monitoring, debugging, and analyzing training progress in the SRLane detection system.