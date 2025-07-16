# Training Runner (`runner.py`)

## File Overview

The Runner class is the central orchestrator for SRLane's training and validation pipeline. It integrates PyTorch Lightning Fabric for distributed training, manages the complete training lifecycle, and coordinates between model, data, optimization, and evaluation components.

## Class: `Runner`

### Initialization (`__init__`)

**Purpose**: Sets up the complete training environment including model, optimization, distributed training, and logging.

**Parameters**:
- `cfg`: Configuration object containing all training parameters

**Key Initialization Steps**:

```python
def __init__(self, cfg):
    # 1. Reproducibility Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    # 2. Core Components
    self.cfg = cfg
    self.recorder = build_recorder(self.cfg)  # Logging and metrics
    self.net = build_net(cfg)                # Model architecture
    self.load_network()                      # Load pretrained weights
    self.optimizer = build_optimizer(self.cfg, self.net)  # Optimizer
    self.scheduler = build_scheduler(self.cfg, self.optimizer)  # LR scheduler
    
    # 3. Distributed Training Setup
    self.fabric = Fabric(
        accelerator="cuda",
        devices=cfg.gpus,
        strategy="dp",          # Data parallel strategy
        precision=cfg.precision
    )
    self.fabric.launch()
    self.net, self.optimizer = self.fabric.setup(self.net, self.optimizer)
    
    # 4. State Variables
    self.val_loader = None
    self.test_loader = None
    self.metric = 0  # Best validation metric
```

**Component Integration**:
- **Recorder**: Handles logging, metrics tracking, and tensorboard integration
- **Network**: Lane detection model (TwoStageDetector)
- **Optimizer**: Parameter optimization (Adam, SGD, etc.)
- **Scheduler**: Learning rate scheduling
- **Fabric**: PyTorch Lightning distributed training wrapper

### Network Loading (`load_network`)

**Purpose**: Loads pretrained weights if specified in configuration.

```python
def load_network(self):
    if not self.cfg.load_from:
        return
    load_network(self.net, self.cfg.load_from, strict=False)
```

**Features**:
- **Conditional Loading**: Only loads if path is specified
- **Non-strict Loading**: Allows partial weight loading for transfer learning
- **Checkpoint Resumption**: Supports resuming from training checkpoints

### Training Methods

#### `train_epoch(train_loader)`

**Purpose**: Executes one complete training epoch with gradient updates and logging.

**Parameters**:
- `train_loader`: DataLoader for training data

**Training Loop Data Flow**:

```python
def train_epoch(self, train_loader):
    self.net.train()  # Set to training mode
    end = time.time()
    
    for i, data in enumerate(train_loader):
        # Early stopping condition
        if self.recorder.step >= self.cfg.total_iter:
            break
            
        # Timing and step tracking
        date_time = time.time() - end
        self.recorder.step += 1
        
        # Forward pass
        output = self.net(data)  # Returns loss and predictions
        
        # Backward pass
        self.optimizer.zero_grad()
        loss = output["loss"].sum()  # Aggregate losses
        self.fabric.backward(loss)   # Distributed backward
        self.optimizer.step()
        self.scheduler.step()
        
        # Timing and logging
        batch_time = time.time() - end
        end = time.time()
        
        # Update metrics
        self.recorder.update_loss_status(output["loss_status"])
        self.recorder.batch_time.update(batch_time)
        self.recorder.data_time.update(date_time)
        
        # Periodic logging
        if i % self.cfg.log_interval == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            self.recorder.lr = lr
            self.recorder.record("train")
```

**Key Features**:
- **Iteration-based Training**: Uses total iterations instead of epochs
- **Distributed Backward**: Leverages Fabric for gradient synchronization
- **Comprehensive Timing**: Tracks both data loading and computation time
- **Loss Aggregation**: Handles multi-component losses from two-stage detector
- **Real-time Logging**: Periodic logging of metrics and learning rate

#### `train()`

**Purpose**: Main training loop that coordinates epochs and validation.

**Training Pipeline**:

```python
def train(self):
    # Setup data loading
    self.recorder.logger.info("Build train_loader...")
    train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)
    train_loader = self.fabric.setup_dataloaders(train_loader)
    
    # Main training loop
    self.recorder.logger.info("Start training...")
    epoch = 0
    while self.recorder.step < self.cfg.total_iter:
        self.recorder.epoch = epoch
        self.train_epoch(train_loader)
        
        # Validation conditions
        if (self.recorder.step >= self.cfg.total_iter or 
            (epoch + 1) % self.cfg.eval_ep == 0):
            self.validate()
        epoch += 1
```

**Control Flow**:
1. **Data Loader Setup**: Creates and configures training data loader
2. **Fabric Integration**: Wraps data loader for distributed training
3. **Epoch Loop**: Continues until iteration limit is reached
4. **Validation Triggers**: Validates at specified intervals or completion

### Validation Method

#### `validate()`

**Purpose**: Performs model evaluation on validation set with metric computation.

**Validation Pipeline**:

```python
@torch.no_grad()
def validate(self):
    # Lazy data loader initialization
    if not self.val_loader:
        self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
    
    net = self.net
    net.eval()  # Set to evaluation mode
    predictions = []
    
    # Inference loop
    for i, data in enumerate(tqdm(self.val_loader, desc="Validate")):
        output = net(data)  # Forward pass
        # Extract lane predictions from ROI head
        output = net.module.roi_head.get_lanes(output, data["meta"])
        predictions.extend(output)
        
        # Optional visualization
        if self.cfg.view:
            self.val_loader.dataset.view(output, data["meta"])
    
    # Evaluation and logging
    metric = self.val_loader.dataset.evaluate(predictions, self.cfg.work_dir)
    self.recorder.logger.info("metric: " + str(metric))
    self.recorder.tb_logger.add_scalar("val/metric", metric, self.recorder.step)
    
    # Best model saving
    if metric > self.metric:
        self.metric = metric
        save_model(net, self.recorder)
```

**Key Features**:
- **No Gradient Computation**: Uses `@torch.no_grad()` decorator for efficiency
- **Lazy Data Loading**: Creates validation loader only when needed
- **Lane Extraction**: Processes model outputs to extract lane predictions
- **Visualization Support**: Optional result visualization during validation
- **Metric Tracking**: Comprehensive evaluation with dataset-specific metrics
- **Best Model Saving**: Automatically saves models with improved performance

## Data Flow and Tensor Shapes

### Training Data Flow
1. **Input Data**: `[B, 3, H, W]` image tensors with metadata
2. **Model Forward**: Produces loss dictionary and intermediate outputs
3. **Loss Aggregation**: Sums component losses from different heads
4. **Gradient Computation**: Distributed backward pass through Fabric
5. **Parameter Update**: Optimizer step with scheduler adjustment

### Validation Data Flow
1. **Input Data**: `[B, 3, H, W]` validation images
2. **Model Inference**: Generates raw predictions
3. **Post-processing**: ROI head extracts lane coordinates
4. **Evaluation**: Dataset-specific metric computation
5. **Logging**: TensorBoard and console output

## Integration with SRLane Architecture

### Component Dependencies
- **Models**: Uses `build_net()` for TwoStageDetector instantiation
- **Datasets**: Integrates with CULane/TuSimple dataset implementations
- **Optimization**: Coordinates optimizer and scheduler through factory functions
- **Logging**: Comprehensive metrics tracking through recorder system

### Configuration Integration
```python
# Key configuration parameters
cfg.total_iter = 80000      # Training iterations
cfg.eval_ep = 5             # Validation frequency (epochs)
cfg.log_interval = 50       # Logging frequency (iterations)
cfg.gpus = [0, 1]          # GPU devices
cfg.precision = "16-mixed"  # Mixed precision training
cfg.seed = 0               # Reproducibility seed
```

### Distributed Training Features
- **Multi-GPU Support**: Data parallel training across multiple GPUs
- **Mixed Precision**: Automatic mixed precision for memory efficiency
- **Gradient Synchronization**: Automatic gradient averaging across devices
- **Data Loading**: Distributed data sampling and loading

The Runner class serves as the central coordinator that orchestrates all aspects of SRLane training, from data loading and model forward passes to optimization and validation, while providing comprehensive logging and distributed training capabilities.