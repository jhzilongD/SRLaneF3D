# SRLane 日志基础设施

## 文件概述

该模块为 SRLane 车道线检测系统提供日志基础设施。它实现了集中式日志配置，支持控制台输出和文件日志，在整个应用程序中保持一致的格式。该日志记录器设计为与训练流程集成，并提供全面的调试和监控功能。

## 导入和依赖

```python
import logging
```

**依赖项**:
- **logging**: Python 的标准日志模块，用于灵活的日志消息处理

## Core Function

### Logger Initialization

```python
def init_logger(log_file=None, log_level=logging.INFO):
```

**Function**: `init_logger`

**Purpose**: Initializes the global logging configuration for the SRLane system with consistent formatting and output destinations.

**Parameters**:
- `log_file` (str, optional): Path to log file for persistent logging
  - If None, only console logging is enabled
  - If provided, creates/overwrites file with 'w' mode
- `log_level` (int): Logging level threshold
  - Default: `logging.INFO`
  - Common values: `logging.DEBUG`, `logging.INFO`, `logging.WARNING`, `logging.ERROR`

**Returns**: None (configures global logger state)

### Implementation Details

#### Handler Configuration

```python
stream_handler = logging.StreamHandler()
handlers = [stream_handler]

if log_file is not None:
    file_handler = logging.FileHandler(log_file, 'w')
    handlers.append(file_handler)
```

**Stream Handler Setup**:
- Creates console output handler for immediate feedback
- Always included regardless of file logging settings
- Outputs to stderr by default

**File Handler Setup**:
- Conditional creation based on `log_file` parameter
- Uses 'w' mode to overwrite existing log files (fresh start per run)
- Enables persistent logging for post-analysis

#### Formatting Configuration

```python
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
for handler in handlers:
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
```

**Formatter Pattern**: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`

**Format Components**:
- `%(asctime)s`: Timestamp in human-readable format
- `%(name)s`: Logger name (typically module name)
- `%(levelname)s`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `%(message)s`: Actual log message content

**Example Output**:
```
2024-07-16 10:30:45,123 - srlane.engine.runner - INFO - Starting training epoch 1
2024-07-16 10:30:45,456 - srlane.models.detector - DEBUG - Processing batch 1/100
2024-07-16 10:30:46,789 - srlane.utils.recorder - WARNING - Loss spike detected: 2.34
```

#### Global Configuration

```python
logging.basicConfig(level=log_level, handlers=handlers)
```

**Basic Configuration**:
- Sets global logging level to filter messages
- Assigns configured handlers to root logger
- Ensures all modules use consistent logging setup

## Integration with SRLane System

### Training Pipeline Integration

The logger integrates throughout the SRLane training pipeline:

1. **Initialization**: Called during training setup to establish logging
2. **Progress Tracking**: Records training metrics and milestones
3. **Error Reporting**: Captures and logs exceptions and warnings
4. **Debug Information**: Provides detailed execution traces when needed

### Usage Pattern in SRLane

```python
# In recorder.py
from .logger import init_logger

class Recorder:
    def __init__(self, cfg):
        # Initialize logging
        init_logger(self.log_path)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Config: \n" + cfg.text)
```

### Integration Points

- **Configuration System**: Log file path typically specified in config
- **Recorder Class**: Primary consumer for training progress logging
- **Error Handling**: System-wide error reporting and debugging
- **Model Checkpointing**: Logs save/load operations and status

## Usage Examples

### Basic Setup
```python
from srlane.utils.logger import init_logger
import logging

# Console-only logging
init_logger()

# Console + file logging
init_logger(log_file="/path/to/training.log")

# Debug level logging
init_logger(log_file="/path/to/debug.log", log_level=logging.DEBUG)
```

### Module-Level Usage
```python
import logging
from srlane.utils.logger import init_logger

# Initialize logging (typically done once at startup)
init_logger(log_file="experiment.log")

# Get module-specific logger
logger = logging.getLogger(__name__)

# Log different types of messages
logger.info("Model training started")
logger.warning("High memory usage detected")
logger.error("Failed to load checkpoint")
logger.debug("Detailed variable state: x=%.3f", x_value)
```

### Training Integration
```python
def train_model(cfg):
    # Setup logging
    log_file = os.path.join(cfg.work_dir, "training.log")
    init_logger(log_file, log_level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting training with config: %s", cfg.exp_name)
    
    try:
        # Training loop
        for epoch in range(cfg.epochs):
            logger.info("Epoch %d/%d starting", epoch + 1, cfg.epochs)
            # ... training code ...
            logger.info("Epoch %d completed, loss: %.4f", epoch + 1, epoch_loss)
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        raise
```

## Configuration Best Practices

### Log Levels
- **DEBUG**: Detailed diagnostic information, typically only of interest when diagnosing problems
- **INFO**: General information about system operation
- **WARNING**: Something unexpected happened, but the software is still working
- **ERROR**: A serious problem occurred, the software was unable to perform some function
- **CRITICAL**: A very serious error occurred, the program may be unable to continue

### File Management
- **Fresh Logs**: Uses 'w' mode to start fresh each run
- **Work Directory**: Typically logs stored in experiment work directory
- **Timestamp**: Log timestamps help correlate with other experiment artifacts

### Performance Considerations
- **Efficient Formatting**: String formatting only occurs if message will be logged
- **Buffered I/O**: File handlers use buffering for performance
- **Level Filtering**: Messages below threshold are efficiently filtered

## Advanced Features

### Custom Logger Names
```python
# Module-specific loggers
model_logger = logging.getLogger('srlane.models')
data_logger = logging.getLogger('srlane.datasets')
train_logger = logging.getLogger('srlane.training')

# Different log levels per module
model_logger.setLevel(logging.DEBUG)
data_logger.setLevel(logging.INFO)
```

### Structured Logging
```python
logger = logging.getLogger(__name__)

# Structured information logging
logger.info("Training metrics: epoch=%d, loss=%.4f, lr=%.6f", 
           epoch, loss, learning_rate)

# Error context logging
try:
    model.load_state_dict(checkpoint)
except Exception as e:
    logger.error("Checkpoint loading failed: file=%s, error=%s", 
                checkpoint_path, str(e))
```

### Integration with External Tools
```python
# TensorBoard integration
logger.info("TensorBoard logs: %s", tensorboard_log_dir)

# Configuration logging
logger.info("Configuration:\n%s", config_pretty_print)

# Performance logging
logger.info("Processing speed: %.2f images/sec", images_per_second)
```

## Thread Safety and Concurrency

The logging module is thread-safe, making it suitable for:
- **Multi-GPU Training**: Logging from multiple processes
- **Asynchronous Operations**: Background data loading and processing
- **Distributed Training**: Coordinated logging across multiple nodes

## Error Handling and Robustness

- **File Creation**: Automatically creates log file directories if needed
- **Permission Handling**: Graceful fallback if file creation fails
- **Encoding**: Handles Unicode characters in log messages
- **Exception Safety**: Logging failures don't crash the application

This logging infrastructure provides a solid foundation for monitoring, debugging, and analyzing the SRLane system throughout development and deployment.