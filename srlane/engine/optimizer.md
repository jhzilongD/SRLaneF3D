# Optimizer Builder (`optimizer.py`)

## File Overview

This module provides a factory function for building PyTorch optimizers from configuration dictionaries. It serves as a bridge between the SRLane configuration system and PyTorch's optimization library, enabling flexible optimizer selection through configuration files.

## Key Functions

### `build_optimizer(cfg, net)`

**Purpose**: Dynamically constructs a PyTorch optimizer based on configuration settings.

**Parameters**:
- `cfg`: Configuration object containing optimizer settings
  - `cfg.optimizer`: Dictionary with optimizer configuration
  - `cfg.optimizer.type`: String specifying the optimizer class name (e.g., "Adam", "SGD")
  - Additional parameters: Learning rate, weight decay, momentum, etc.
- `net`: PyTorch neural network model whose parameters will be optimized

**Returns**: Instantiated PyTorch optimizer object

**Data Flow**:
1. **Configuration Parsing**: Creates a copy of the optimizer configuration to avoid modifying the original
2. **Type Extraction**: Extracts the optimizer type (class name) from the configuration
3. **Validation**: Checks if the specified optimizer type exists in `torch.optim`
4. **Dynamic Instantiation**: Uses `getattr()` to retrieve the optimizer class and instantiates it
5. **Parameter Binding**: Passes the network's parameters and configuration to the optimizer

```python
def build_optimizer(cfg, net):
    cfg_cp = cfg.optimizer.copy()        # Avoid modifying original config
    cfg_type = cfg_cp.pop("type")        # Extract optimizer type
    
    if cfg_type not in dir(torch.optim):  # Validate optimizer exists
        raise ValueError(f"{cfg_type} is not defined.")
    
    _optim = getattr(torch.optim, cfg_type)  # Get optimizer class
    return _optim(net.parameters(), **cfg_cp)  # Instantiate with parameters
```

## Configuration Integration

The function expects configuration in the following format:

```python
cfg.optimizer = {
    "type": "Adam",           # Optimizer class name
    "lr": 0.001,             # Learning rate
    "weight_decay": 1e-4,    # Weight decay
    "betas": [0.9, 0.999]    # Adam-specific parameters
}
```

## Supported Optimizers

All PyTorch optimizers in `torch.optim` are supported, including:
- **Adam**: Adaptive learning rate optimizer
- **SGD**: Stochastic Gradient Descent
- **AdamW**: Adam with decoupled weight decay
- **RMSprop**: Root Mean Square propagation
- **Adamax**: Adam variant with infinity norm

## Error Handling

**ValueError**: Raised when the specified optimizer type doesn't exist in PyTorch's optimizer library. This prevents runtime errors and provides clear feedback about configuration issues.

## Usage in SRLane

The optimizer builder is used in the `Runner` class during training initialization:

```python
# In runner.py
self.optimizer = build_optimizer(self.cfg, self.net)
```

This allows experiments to be configured entirely through config files without code changes:

```python
# In config file
optimizer = dict(
    type='Adam',
    lr=1e-3,
    weight_decay=1e-4
)
```

## Design Patterns

1. **Factory Pattern**: Creates objects based on configuration without requiring explicit class imports
2. **Configuration-Driven**: Enables switching optimizers through configuration changes alone
3. **Parameter Forwarding**: Uses `**kwargs` to forward arbitrary optimizer-specific parameters
4. **Defensive Copying**: Copies configuration to avoid side effects on the original config object

## Integration with Training Pipeline

The built optimizer integrates with:
- **Lightning Fabric**: For distributed training setup
- **Scheduler**: Learning rate scheduling based on training progress
- **Gradient Updates**: Called during training loop for parameter updates
- **Checkpointing**: State is saved/loaded for training resumption