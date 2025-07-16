# SRLane Dataset Registry System

## File Overview
This file implements the registry system for datasets and data processing components in SRLane. It provides factory functions for creating datasets and dataloaders, along with worker initialization for reproducible training across multiple processes.

## Code Structure

### Registry Definitions
```python
DATASETS = Registry("datasets")
PROCESS = Registry("process")
```

#### Purpose:
- **`DATASETS`**: Registry for dataset implementations (e.g., CULane, TuSimple)
- **`PROCESS`**: Registry for data processing and augmentation modules

These registries enable modular and configurable component creation throughout the SRLane system.

### Build Function
```python
def build(cfg, registry, default_args=None):
```

#### Purpose:
Generic factory function that creates objects from configuration dictionaries using the specified registry.

#### Parameters:
- **`cfg`** (dict or list): Configuration object(s) to build from
- **`registry`** (Registry): Registry containing the target component type
- **`default_args`** (dict, optional): Default arguments to pass to constructors

#### Logic:
- **List Handling**: For list configurations, builds each component and wraps in `torch.nn.Sequential`
- **Single Object**: For dict configurations, builds single component using `build_from_cfg`

#### Returns:
- **Sequential**: For list inputs, returns `torch.nn.Sequential` containing all components
- **Single Object**: For dict inputs, returns the constructed component

### Dataset Builder
```python
def build_dataset(split_cfg, cfg):
```

#### Purpose:
Factory function specifically for creating dataset instances.

#### Parameters:
- **`split_cfg`** (dict): Configuration for the dataset split
- **`cfg`** (object): Global configuration object passed as default argument

#### Usage Pattern:
```python
dataset = build_dataset(
    split_cfg={"type": "CULane", "split": "train", ...},
    cfg=global_config
)
```

#### Integration:
- Uses `DATASETS` registry to instantiate the correct dataset class
- Passes global configuration as default argument to dataset constructor

### Worker Initialization
```python
def worker_init_fn(worker_id, seed):
```

#### Purpose:
Ensures reproducible random number generation across DataLoader worker processes.

#### Parameters:
- **`worker_id`** (int): Unique identifier for the worker process
- **`seed`** (int): Base random seed from global configuration

#### Reproducibility Strategy:
1. **Worker-Specific Seed**: `worker_seed = worker_id + seed`
2. **NumPy Seeding**: `np.random.seed(worker_seed)`
3. **Python Seeding**: `random.seed(worker_seed)`

#### Benefits:
- Ensures deterministic behavior across multiple training runs
- Maintains randomness while providing reproducibility
- Prevents worker processes from having identical random states

### DataLoader Builder
```python
def build_dataloader(split_cfg, cfg, is_train=True):
```

#### Purpose:
Factory function for creating PyTorch DataLoader instances with proper configuration for SRLane training and evaluation.

#### Parameters:
- **`split_cfg`** (dict): Dataset split configuration
- **`cfg`** (object): Global configuration containing training parameters
- **`is_train`** (bool): Whether this is a training dataloader (affects shuffling)

#### Configuration Extraction:
- **`samples_per_gpu`**: Calculated as `cfg.batch_size // cfg.gpus`
- **`batch_size`**: Total batch size across all GPUs
- **`num_workers`**: Number of worker processes for data loading
- **`shuffle`**: Enabled for training, disabled for validation/testing

#### DataLoader Configuration:
```python
data_loader = torch.utils.data.DataLoader(
    dataset=built_dataset,
    batch_size=cfg.batch_size,
    shuffle=is_train,
    num_workers=cfg.workers,
    pin_memory=False,
    drop_last=False,
    collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
    worker_init_fn=partial(worker_init_fn, seed=cfg.seed)
)
```

#### Key Features:
- **Custom Collation**: Uses MMEngine's `collate` function with GPU-aware batching
- **Reproducible Workers**: Applies `worker_init_fn` for deterministic behavior
- **Memory Management**: `pin_memory=False` for compatibility across systems
- **Batch Handling**: `drop_last=False` to use all available data

## Data Flow and Integration

### Dataset Creation Flow:
```
Configuration → build_dataset() → Registry Lookup → Dataset.__init__() → Dataset Instance
```

### DataLoader Creation Flow:
```
Configuration → build_dataloader() → build_dataset() → DataLoader Configuration → DataLoader Instance
```

### Worker Process Flow:
```
DataLoader Start → Worker Creation → worker_init_fn() → Seeded Worker Process → Data Loading
```

## Configuration Dependencies

### Required Configuration Parameters:
- **`batch_size`** (int): Total batch size across all GPUs
- **`gpus`** (int): Number of GPUs for training
- **`workers`** (int): Number of DataLoader worker processes
- **`seed`** (int): Random seed for reproducibility

### Dataset Configuration Format:
```python
split_cfg = {
    "type": "CULane",          # Dataset class name (registered)
    "data_root": "/path/to/data",
    "split": "train",
    "processes": [...],         # Data processing pipeline
}
```

## Usage Examples

### Basic Dataset Creation:
```python
# Configuration
dataset_config = {
    "type": "CULane",
    "data_root": "/data/culane",
    "split": "train"
}

# Create dataset
dataset = build_dataset(dataset_config, global_config)
```

### DataLoader Creation:
```python
# Create training dataloader
train_loader = build_dataloader(
    split_cfg=train_config,
    cfg=global_config,
    is_train=True
)

# Create validation dataloader  
val_loader = build_dataloader(
    split_cfg=val_config,
    cfg=global_config,
    is_train=False
)
```

### Custom Component Registration:
```python
@DATASETS.register_module
class CustomDataset(BaseDataset):
    def __init__(self, ...):
        # Custom dataset implementation
        pass

@PROCESS.register_module  
class CustomTransform:
    def __call__(self, sample):
        # Custom processing logic
        return processed_sample
```

## Multi-GPU Considerations

### Batch Distribution:
- **Total Batch Size**: Specified in configuration
- **Per-GPU Batch Size**: Automatically calculated as `batch_size // gpus`
- **Collation Function**: Handles proper batching for distributed training

### Worker Process Management:
- **Reproducibility**: Each worker gets unique but deterministic seed
- **Resource Allocation**: Worker count configured based on system capabilities
- **Memory Efficiency**: Pin memory settings optimized for multi-GPU setups

## Integration Points

### Training Pipeline:
- Used by training engines to create data loaders
- Integrates with Lightning-based training systems
- Supports both single and multi-GPU training configurations

### Configuration System:
- Works seamlessly with MMEngine configuration framework
- Enables declarative dataset and processing specification
- Supports hierarchical configuration inheritance

### Evaluation Pipeline:
- Creates evaluation data loaders with appropriate settings
- Ensures consistent data loading between training and validation
- Supports different evaluation protocols per dataset

## Performance Optimization

### Efficient Data Loading:
- Configurable number of worker processes
- Optimized collation for batch processing
- Memory pinning options for GPU transfer

### Reproducibility vs Performance:
- Balanced approach to random seeding
- Worker-specific seeds prevent correlation
- Deterministic behavior without sacrificing randomness

This registry system provides a flexible and scalable foundation for managing datasets and data processing in the SRLane framework, enabling easy extension and configuration while maintaining performance and reproducibility.