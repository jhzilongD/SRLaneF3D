# Component Registry System (`registry.py`)

## File Overview

This module implements a registry-based component system for SRLane, enabling dynamic instantiation of trainers and evaluators through configuration. It leverages the MMEngine registry pattern to provide a flexible, extensible architecture for different training and evaluation strategies.

## Key Components

### Global Registries

```python
TRAINER = Registry("trainer")      # Registry for training components
EVALUATOR = Registry("evaluator")  # Registry for evaluation components
```

**Purpose**: Central repositories for registering and retrieving trainer and evaluator implementations.

**Registry Pattern Benefits**:
- **Decoupling**: Separates component definition from instantiation
- **Extensibility**: New components can be added without modifying core code
- **Configuration-Driven**: Components selected through config files

### Core Functions

#### `build(cfg, registry, default_args=None)`

**Purpose**: Generic builder function that can handle both single components and sequential chains.

**Parameters**:
- `cfg`: Configuration dictionary or list of configurations
- `registry`: Target registry (TRAINER or EVALUATOR)
- `default_args`: Default arguments passed to all built components

**Return Types**:
- **Single Component**: Direct instance when `cfg` is a dictionary
- **Sequential Chain**: `nn.Sequential` container when `cfg` is a list

**Data Flow**:
```python
def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        # Build multiple components and chain them
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)  # Chain components sequentially
    else:
        # Build single component
        return build_from_cfg(cfg, registry, default_args)
```

**Use Cases**:
- **Single Component**: Building a single trainer or evaluator
- **Component Chains**: Creating pipelines of multiple trainers/evaluators
- **Complex Workflows**: Sequentially applying different training strategies

#### `build_trainer(cfg)`

**Purpose**: Specialized builder for training components.

**Parameters**:
- `cfg`: Complete configuration object containing trainer specification

**Implementation**:
```python
def build_trainer(cfg):
    return build(cfg.trainer, TRAINER, default_args=dict(cfg=cfg))
```

**Data Flow**:
1. **Configuration Extraction**: Extracts trainer config from main configuration
2. **Default Arguments**: Passes full configuration as default argument
3. **Registry Lookup**: Uses TRAINER registry for component resolution
4. **Instantiation**: Creates trainer instance with configuration

#### `build_evaluator(cfg)`

**Purpose**: Specialized builder for evaluation components.

**Parameters**:
- `cfg`: Complete configuration object containing evaluator specification

**Implementation**:
```python
def build_evaluator(cfg):
    return build(cfg.evaluator, EVALUATOR, default_args=dict(cfg=cfg))
```

**Features**:
- **Consistent Interface**: Mirrors trainer builder pattern
- **Configuration Access**: Evaluators receive full config for complex evaluation logic
- **Registry Isolation**: Uses separate EVALUATOR registry

## Configuration Integration

### Trainer Configuration
```python
cfg.trainer = {
    "type": "DefaultTrainer",
    "max_epochs": 100,
    "validation_interval": 5
}
```

### Evaluator Configuration
```python
cfg.evaluator = {
    "type": "CULaneEvaluator",
    "metric_types": ["F1", "accuracy"],
    "output_dir": "eval_results"
}
```

### Sequential Configuration
```python
cfg.trainer = [
    {"type": "WarmupTrainer", "warmup_epochs": 5},
    {"type": "MainTrainer", "epochs": 95}
]
```

## Component Registration

Components must be registered before use:

```python
@TRAINER.register_module()
class CustomTrainer:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        # Implementation...

@EVALUATOR.register_module()
class CustomEvaluator:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        # Implementation...
```

## Error Handling

The registry system provides robust error handling:
- **Missing Components**: Clear error messages when components aren't registered
- **Configuration Validation**: Type checking and parameter validation
- **Import Resolution**: Automatic module importing for registered components

## Usage in SRLane Architecture

### Training Pipeline Integration
```python
# In runner or training script
trainer = build_trainer(cfg)
evaluator = build_evaluator(cfg)

# Use in training loop
for epoch in range(cfg.epochs):
    trainer.train_epoch(model, dataloader)
    if epoch % cfg.eval_interval == 0:
        metrics = evaluator.evaluate(model, val_dataloader)
```

### Extension Points

The registry system enables easy extension:

1. **Custom Trainers**: Different training strategies (adversarial, meta-learning)
2. **Custom Evaluators**: Domain-specific evaluation metrics
3. **Training Pipelines**: Sequential training phases
4. **A/B Testing**: Easy switching between trainer implementations

## Design Patterns

1. **Registry Pattern**: Central registration of components
2. **Factory Pattern**: Dynamic component creation
3. **Builder Pattern**: Step-by-step component construction
4. **Strategy Pattern**: Interchangeable trainer/evaluator implementations

## Dependencies

- **MMEngine Registry**: Underlying registry implementation
- **PyTorch nn.Sequential**: For chaining components
- **build_from_cfg**: MMEngine utility for configuration-based building

The registry system forms the backbone of SRLane's modular architecture, enabling flexible experimentation and easy extension of training and evaluation capabilities.