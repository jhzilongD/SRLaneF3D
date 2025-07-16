# Learning Rate Scheduler Builder (`scheduler.py`)

## File Overview

This module provides a factory function for building PyTorch learning rate schedulers from configuration dictionaries. It extends PyTorch's standard scheduler library with a custom warmup scheduler that combines linear warmup with cosine annealing, which is particularly effective for training deep lane detection models.

## Key Functions

### `build_scheduler(cfg, optimizer)`

**Purpose**: Dynamically constructs learning rate schedulers based on configuration, with special support for warmup schedules.

**Parameters**:
- `cfg`: Configuration object containing scheduler settings
  - `cfg.scheduler`: Dictionary with scheduler configuration
  - `cfg.scheduler.type`: String specifying scheduler type ("warmup", "StepLR", "CosineAnnealingLR", etc.)
- `optimizer`: PyTorch optimizer instance to be scheduled

**Returns**: Instantiated PyTorch learning rate scheduler

**Implementation Flow**:

```python
def build_scheduler(cfg, optimizer):
    cfg_cp = cfg.scheduler.copy()          # Avoid modifying original config
    cfg_type = cfg_cp.pop("type")          # Extract scheduler type
    
    # Validation for non-warmup schedulers
    if cfg_type != "warmup" and cfg_type not in dir(torch.optim.lr_scheduler):
        raise ValueError(f"{cfg_type} is not defined.")
    
    # Custom warmup scheduler implementation
    if cfg_type == "warmup":
        def warm_up_cosine_lr(iteration):
            warm_up = cfg_cp["warm_up_iters"]
            if iteration <= warm_up:
                # Linear warmup phase
                return iteration / warm_up
            else:
                # Cosine annealing phase
                return 0.5 * (math.cos((iteration - warm_up) / (
                       cfg_cp["total_iters"] - warm_up) * math.pi) + 1)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_cosine_lr)
    
    # Standard PyTorch schedulers
    _scheduler = getattr(torch.optim.lr_scheduler, cfg_type)
    return _scheduler(optimizer, **cfg_cp)
```

## Custom Warmup Scheduler

### `warm_up_cosine_lr(iteration)`

**Purpose**: Implements a learning rate schedule that combines linear warmup with cosine annealing.

**Parameters**:
- `iteration`: Current training iteration (0-indexed)

**Returns**: Learning rate multiplier (0.0 to 1.0)

**Mathematical Formula**:

**Warmup Phase** (iterations 0 to `warm_up_iters`):
```
lr_multiplier = iteration / warm_up_iters
```

**Cosine Annealing Phase** (iterations `warm_up_iters` to `total_iters`):
```
progress = (iteration - warm_up_iters) / (total_iters - warm_up_iters)
lr_multiplier = 0.5 * (cos(progress * π) + 1)
```

### Learning Rate Curve Analysis

The warmup scheduler produces a characteristic learning rate curve:

1. **Linear Warmup**: Gradually increases from 0 to base learning rate
   - Prevents gradient explosion in early training
   - Allows for stable initialization of batch normalization statistics
   - Duration: `warm_up_iters` iterations

2. **Cosine Annealing**: Smoothly decreases from base learning rate to near zero
   - Provides fine-tuning capability in later training stages
   - Smooth transitions prevent training instability
   - Duration: `total_iters - warm_up_iters` iterations

### Configuration Example

```python
# Warmup scheduler configuration
cfg.scheduler = {
    "type": "warmup",
    "warm_up_iters": 1000,      # Linear warmup for first 1000 iterations
    "total_iters": 80000        # Total training iterations
}

# Standard scheduler configuration
cfg.scheduler = {
    "type": "StepLR",
    "step_size": 30,            # Decay every 30 epochs
    "gamma": 0.1                # Multiply by 0.1
}
```

## Supported Schedulers

### Custom Schedulers
- **warmup**: Linear warmup + cosine annealing combination

### PyTorch Standard Schedulers
- **StepLR**: Decay learning rate by gamma every step_size epochs
- **MultiStepLR**: Decay at specific milestones
- **ExponentialLR**: Exponential decay
- **CosineAnnealingLR**: Cosine annealing without warmup
- **ReduceLROnPlateau**: Reduce on metric plateau
- **CyclicLR**: Cyclical learning rates
- **OneCycleLR**: One cycle learning rate policy

## Error Handling

**ValueError**: Raised when the specified scheduler type doesn't exist in PyTorch's scheduler library (excluding the custom "warmup" type).

## Usage in SRLane Training

The scheduler builder integrates with the training pipeline:

```python
# In runner.py
self.scheduler = build_scheduler(self.cfg, self.optimizer)

# In training loop
for iteration in range(total_iterations):
    # Forward pass and backward pass
    output = self.net(data)
    loss = output["loss"].sum()
    self.fabric.backward(loss)
    self.optimizer.step()
    self.scheduler.step()  # Update learning rate
```

## Benefits of Warmup Scheduling

### For Lane Detection Training

1. **Stability**: Prevents early training instability common in complex multi-stage models
2. **Convergence**: Improves final model performance through better optimization trajectory
3. **Generalization**: Cosine annealing provides implicit regularization
4. **Efficiency**: Allows for higher peak learning rates with stable training

### Training Dynamics

**Early Training** (Warmup Phase):
- Model learns basic feature representations
- Batch normalization statistics stabilize
- Prevents gradient explosion in deep networks

**Mid Training** (High Learning Rate):
- Rapid convergence to good solutions
- Explores parameter space effectively
- Makes significant progress on loss reduction

**Late Training** (Annealing Phase):
- Fine-tunes model parameters
- Converges to local minimum
- Improves generalization performance

## Mathematical Properties

### Derivatives and Smoothness
The cosine function provides smooth learning rate transitions:
- Continuous first derivatives prevent learning rate jumps
- Gradual transitions maintain training stability
- Natural decay profile matches optimization theory

### Iteration-Based Scheduling
Unlike epoch-based scheduling, iteration-based scheduling provides:
- Finer control over learning rate changes
- Consistency across different dataset sizes
- Better synchronization with validation timing

## Integration with Distributed Training

The scheduler works seamlessly with PyTorch Lightning Fabric:
- Learning rate updates are synchronized across devices
- Iteration counting remains consistent in multi-GPU training
- Scheduler state is properly saved/loaded in checkpoints

The scheduler builder provides a crucial component for stable and effective training of the SRLane lane detection model, enabling both standard scheduling strategies and advanced warmup techniques optimized for deep learning applications.