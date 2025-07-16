# Speed Measurement Tool (`speed_measure.py`)

## File Overview

A specialized benchmarking tool for measuring inference speed of SRLane lane detection models. This script provides accurate performance measurements by implementing proper GPU warming, synchronization, and statistical timing to evaluate model efficiency for real-time applications.

## Purpose

The speed measurement tool is crucial for:
- **Performance Benchmarking**: Evaluating model efficiency for deployment
- **Hardware Optimization**: Comparing performance across different GPU configurations
- **Model Comparison**: Quantitative speed comparisons between model variants
- **Real-time Feasibility**: Determining if models meet real-time requirements (e.g., 30+ FPS)

## Key Functions

### `parse_args()`

**Purpose**: Parses command-line arguments for speed measurement configuration.

**Returns**: `argparse.Namespace` with benchmark parameters

**Arguments**:
```python
def parse_args():
    parser = argparse.ArgumentParser(description="Speed measure")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--repetitions", default=1000, type=int,
                        help="Repeat times")
    parser.add_argument("--warmup", default=200, type=int,
                        help="Trigger GPU initialization")
    args = parser.parse_args()
    args.cuda = True  # Force CUDA usage for speed measurement
    return args
```

**Parameter Details**:
- **config**: Path to model configuration file
- **--repetitions**: Number of inference runs for averaging (default: 1000)
- **--warmup**: Number of warmup iterations to stabilize GPU (default: 200)
- **cuda**: Automatically set to True for GPU benchmarking

### `main()`

**Purpose**: Executes the complete speed measurement pipeline with proper GPU synchronization.

**Benchmarking Pipeline**:

```python
@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    cfg = Config.fromfile(args.config)
    torch.backends.cudnn.benchmark = False  # Disable for consistent timing
    
    # 1. Model Setup
    net = build_net(cfg)
    print(net)  # Display model architecture
    net = net.to(device)
    net.eval()  # Set to evaluation mode
    
    # 2. GPU Warmup Phase
    for i in range(args.warmup):
        input = torch.zeros(1, 3, cfg.img_h, cfg.img_w)
        input = input.to(device)
        net(input)  # Warmup run
    
    # 3. Benchmarking Phase
    input = torch.zeros(1, 3, cfg.img_h, cfg.img_w)
    input = input.to(device)
    
    if args.cuda:
        torch.cuda.current_stream(device).synchronize()
    
    start = time.perf_counter()
    for i in range(args.repetitions):
        net(input)
    
    if args.cuda:
        torch.cuda.current_stream(device).synchronize()
    
    sum_time = (time.perf_counter() - start) * 1000
    
    # 4. Results
    print(f"avg time = {sum_time / args.repetitions:.2f}ms")
```

## Benchmarking Methodology

### GPU Warmup Protocol

**Purpose**: Eliminates GPU initialization overhead and stabilizes performance.

```python
for i in range(args.warmup):
    input = torch.zeros(1, 3, cfg.img_h, cfg.img_w)
    input = input.to(device)
    net(input)
```

**Why Warmup is Critical**:
1. **GPU Initialization**: First runs include CUDA context creation overhead
2. **Memory Allocation**: Initial tensor allocations are slower
3. **Kernel Compilation**: cuDNN selects and compiles optimal kernels
4. **Thermal Stabilization**: GPU reaches stable operating temperature

**Warmup Iterations**: 200 iterations ensure complete stabilization

### Precise Timing Protocol

**High-Resolution Timer**:
```python
start = time.perf_counter()  # Nanosecond precision
# ... benchmark iterations ...
sum_time = (time.perf_counter() - start) * 1000  # Convert to milliseconds
```

**GPU Synchronization**:
```python
if args.cuda:
    torch.cuda.current_stream(device).synchronize()
```

**Why Synchronization Matters**:
- GPU operations are asynchronous by default
- CPU timing without sync measures kernel launch time, not execution time
- Synchronization ensures accurate measurement of actual computation time

### Statistical Reliability

**Multiple Iterations**:
- Default: 1000 repetitions for robust averaging
- Reduces impact of system noise and scheduling variations
- Provides statistically significant timing measurements

**Consistent Environment**:
```python
torch.backends.cudnn.benchmark = False
```
Disables cuDNN autotuner to ensure consistent kernel selection across runs.

## Input Configuration

### Tensor Shape Extraction
```python
input = torch.zeros(1, 3, cfg.img_h, cfg.img_w)
```

**Dimensions**:
- **Batch Size**: 1 (single image inference)
- **Channels**: 3 (RGB color channels)
- **Height/Width**: Extracted from configuration (`cfg.img_h`, `cfg.img_w`)

**Common SRLane Input Sizes**:
- CULane: 288×800 pixels
- TuSimple: 288×800 pixels
- Custom datasets: Configurable dimensions

### Memory Layout
- **Data Type**: float32 (default PyTorch tensor type)
- **Memory Location**: GPU memory for CUDA benchmarking
- **Layout**: NCHW format (batch, channels, height, width)

## Usage Examples

### Basic Speed Measurement
```bash
python tools/analysis_tools/speed_measure.py configs/exp_srlane_culane.py
```

### Custom Repetitions
```bash
python tools/analysis_tools/speed_measure.py configs/exp_srlane_culane.py \
    --repetitions 2000 \
    --warmup 300
```

### Output Analysis
```
avg time = 12.45ms
```
This indicates the model can process images at approximately 80 FPS (1000ms / 12.45ms).

## Performance Interpretation

### Real-time Thresholds
- **30 FPS**: ≤ 33.33ms per frame
- **60 FPS**: ≤ 16.67ms per frame
- **100 FPS**: ≤ 10.00ms per frame

### Factors Affecting Speed
1. **Model Architecture**: Number of parameters and operations
2. **Input Resolution**: Higher resolution increases computation
3. **GPU Hardware**: Memory bandwidth and compute capability
4. **Batch Size**: Larger batches improve throughput but increase latency
5. **Precision**: FP16 vs FP32 affects speed and memory usage

## Integration with SRLane

### Model Architecture Impact
The speed measurement evaluates the complete SRLane pipeline:
- **Backbone**: ResNet feature extraction
- **Neck**: Feature pyramid processing
- **RPN Head**: Initial lane proposal generation
- **Refinement Head**: Multi-stage lane refinement
- **Post-processing**: Lane coordinate extraction

### Configuration Compatibility
Works with all SRLane model configurations:
- Different backbone architectures (ResNet18, ResNet34, ResNet50)
- Various input resolutions
- Different neck configurations
- Custom head configurations

## Debugging and Optimization

### Memory Usage Monitoring
```python
print(torch.cuda.memory_allocated() / 1024**2)  # MB
print(torch.cuda.memory_reserved() / 1024**2)   # MB
```

### Bottleneck Identification
- Compare with other models to identify relative performance
- Profile individual components using PyTorch profiler
- Analyze GPU utilization during inference

## Design Considerations

### Accuracy vs Speed Trade-offs
- **No Gradient Computation**: Uses `@torch.no_grad()` for efficiency
- **Evaluation Mode**: Disables dropout and batch norm updates
- **Single Image Batch**: Measures per-image latency, not throughput

### Reproducibility
- Fixed random seeds not needed (no randomness in inference)
- Consistent GPU state through warmup
- Deterministic timing through proper synchronization

The speed measurement tool provides essential performance metrics for deploying SRLane models in real-time applications, ensuring that lane detection systems can meet the stringent timing requirements of autonomous driving and ADAS applications.