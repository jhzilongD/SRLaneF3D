# Main Training Script (`main.py`)

## File Overview

The main entry point for SRLane training and validation. This script provides a command-line interface for running experiments, handling configuration parsing, GPU setup, and coordination between training and validation modes. It serves as the primary interface for users to interact with the SRLane system.

## Key Functions

### `parse_args()`

**Purpose**: Parses command-line arguments for training and validation configuration.

**Returns**: `argparse.Namespace` object with parsed arguments

**Supported Arguments**:

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    
    # Required argument
    parser.add_argument("config", help="Config file path")
    
    # Optional arguments
    parser.add_argument("--work_dirs", type=str, default=None,
                        help="Dirs for log and saving ckpts")
    parser.add_argument("--load_from", default=None,
                        help="The checkpoint file to load from")
    parser.add_argument("--view", action="store_true",
                        help="Whether to visualize results during validation")
    parser.add_argument("--validate", action="store_true",
                        help="Whether to evaluate the checkpoint")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0, ],
                        help="Used GPU indices")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
```

**Argument Details**:

- **config** (required): Path to configuration file (e.g., `configs/exp_srlane_culane.py`)
- **--work_dirs**: Custom output directory for logs and checkpoints
- **--load_from**: Path to pretrained checkpoint for initialization or validation
- **--view**: Enables visualization of lane detection results during validation
- **--validate**: Switches to validation-only mode (no training)
- **--gpus**: List of GPU indices to use (supports multi-GPU training)
- **--seed**: Random seed for reproducible experiments

### `main()`

**Purpose**: Main execution function that orchestrates the training/validation pipeline.

**Execution Flow**:

```python
def main():
    args = parse_args()
    
    # 1. GPU Environment Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu) for gpu in args.gpus)
    
    # 2. Configuration Loading and Merging
    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)
    cfg.load_from = args.load_from
    cfg.view = args.view
    cfg.seed = args.seed
    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs
    
    # 3. Performance Optimization
    cudnn.benchmark = True
    
    # 4. Runner Initialization and Execution
    runner = Runner(cfg)
    if args.validate:
        runner.validate()
    else:
        runner.train()
```

## Configuration Integration

### Configuration Loading
Uses MMEngine's `Config.fromfile()` to load Python-based configuration files:

```python
cfg = Config.fromfile(args.config)
```

**Supported Config Formats**:
- Python files with configuration dictionaries
- Hierarchical configuration inheritance
- Dynamic configuration generation

### Configuration Override
Command-line arguments override configuration file values:

```python
cfg.gpus = len(args.gpus)          # Number of GPUs
cfg.load_from = args.load_from     # Checkpoint path
cfg.view = args.view               # Visualization flag
cfg.seed = args.seed               # Random seed
cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs
```

This allows flexible experimentation without modifying configuration files.

## GPU Management

### Environment Variable Setup
```python
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
```

**Purpose**: Controls which GPUs are visible to PyTorch
**Benefits**:
- Isolates training to specific GPUs
- Enables multi-GPU training on selected devices
- Prevents interference with other processes

### Performance Optimization
```python
cudnn.benchmark = True
```

**Purpose**: Enables cuDNN autotuner for optimal convolution algorithms
**Benefits**:
- Automatically selects fastest convolution implementations
- Improves training speed for fixed input sizes
- Optimizes for specific hardware configurations

## Usage Examples

### Training Examples

**Single GPU Training**:
```bash
python tools/main.py configs/exp_srlane_culane.py --gpus 0
```

**Multi-GPU Training**:
```bash
python tools/main.py configs/exp_srlane_culane.py --gpus 0 1 2 3
```

**Training with Custom Settings**:
```bash
python tools/main.py configs/exp_srlane_culane.py \
    --gpus 0 1 \
    --work_dirs ./custom_output \
    --seed 42
```

**Resume Training from Checkpoint**:
```bash
python tools/main.py configs/exp_srlane_culane.py \
    --load_from ./work_dirs/ckpt/best_model.pth \
    --gpus 0
```

### Validation Examples

**Standard Validation**:
```bash
python tools/main.py configs/exp_srlane_culane.py \
    --load_from checkpoint/baseline.pth \
    --validate \
    --gpus 0
```

**Validation with Visualization**:
```bash
python tools/main.py configs/exp_srlane_culane.py \
    --load_from checkpoint/baseline.pth \
    --validate \
    --view \
    --gpus 0
```

## Mode Selection

### Training Mode (Default)
- Executed when `--validate` flag is not provided
- Runs complete training pipeline with periodic validation
- Saves checkpoints and logs training progress
- Automatically validates at specified intervals

### Validation Mode
- Activated with `--validate` flag
- Loads model from checkpoint specified by `--load_from`
- Runs inference on validation set
- Computes and reports evaluation metrics
- Optionally visualizes results with `--view` flag

## Error Handling and Validation

### Required Arguments
- Script validates that configuration file path is provided
- Exits with usage message if required arguments are missing

### Configuration Validation
- MMEngine Config system validates configuration file syntax
- Reports detailed error messages for configuration issues

### GPU Availability
- Relies on PyTorch CUDA availability checks
- cuDNN benchmark optimization is conditionally applied

## Integration with SRLane Architecture

### Component Coordination
The main script coordinates several SRLane components:

1. **Configuration System**: Loads and merges experimental settings
2. **Runner Class**: Delegates execution to training/validation orchestrator
3. **GPU Management**: Sets up distributed training environment
4. **Logging**: Inherits logging configuration from Runner

### Experiment Management
The script supports various experimental workflows:

- **Hyperparameter Sweeps**: Different configs with same script
- **Ablation Studies**: Modifying specific configuration components
- **Transfer Learning**: Loading pretrained weights with `--load_from`
- **Performance Analysis**: Validation-only runs for model evaluation

## Design Patterns

1. **Command-Line Interface**: Standard argparse pattern for user interaction
2. **Configuration Override**: Command-line arguments take precedence over config files
3. **Mode Selection**: Single script handles both training and validation
4. **Environment Setup**: Systematic GPU and performance configuration
5. **Delegation**: Core logic delegated to specialized Runner class

The main script provides a clean, flexible interface for running SRLane experiments while handling the complexities of GPU management, configuration loading, and mode selection behind a simple command-line interface.