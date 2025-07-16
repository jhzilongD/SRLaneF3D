# 优化器构建器 (`optimizer.py`)

## 文件概述

该模块提供了从配置字典构建PyTorch优化器的工厂函数。它作为SRLane配置系统和PyTorch优化库之间的桥梁，通过配置文件实现灵活的优化器选择。

## 关键函数

### `build_optimizer(cfg, net)`

**作用**: 基于配置设置动态构建 PyTorch 优化器。

**参数**:
- `cfg`: 包含优化器设置的配置对象
  - `cfg.optimizer`: 优化器配置字典
  - `cfg.optimizer.type`: 指定优化器类名的字符串（例如，"Adam", "SGD"）
  - 附加参数: 学习率、权重衰减、动量等
- `net`: 其参数将被优化的 PyTorch 神经网络模型

**返回**: 实例化的 PyTorch 优化器对象

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