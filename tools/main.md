# 主训练脚本 (`main.py`)

## 文件概述

SRLane 训练和验证的主入口点。该脚本提供了运行实验的命令行界面，处理配置解析、GPU 设置以及训练和验证模式之间的协调。它是用户与 SRLane 系统交互的主要接口。

## 关键函数

### `parse_args()`

**作用**: 解析训练和验证配置的命令行参数。

**返回**: 包含解析参数的 `argparse.Namespace` 对象

**支持的参数**:

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    
    # 必需参数
    parser.add_argument("config", help="配置文件路径")
    
    # 可选参数
    parser.add_argument("--work_dirs", type=str, default=None,
                        help="日志和保存检查点的目录")
    parser.add_argument("--load_from", default=None,
                        help="要加载的检查点文件")
    parser.add_argument("--view", action="store_true",
                        help="验证期间是否可视化结果")
    parser.add_argument("--validate", action="store_true",
                        help="是否评估检查点")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0, ],
                        help="使用的 GPU 索引")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子")
```

**参数详情**:

- **config** (必需): 配置文件路径（例如，`configs/exp_srlane_culane.py`）
- **--work_dirs**: 自定义日志和检查点输出目录
- **--load_from**: 用于初始化或验证的预训练检查点路径
- **--view**: 在验证期间启用车道线检测结果的可视化
- **--validate**: 切换到仅验证模式（不训练）
- **--gpus**: 要使用的 GPU 索引列表（支持多 GPU 训练）
- **--seed**: 可重现实验的随机种子

### `main()`

**作用**: 统筹训练/验证流程的主执行函数。

**执行流程**:

```python
def main():
    args = parse_args()
    
    # 1. GPU 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu) for gpu in args.gpus)
    
    # 2. 配置加载和合并
    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)
    cfg.load_from = args.load_from
    cfg.view = args.view
    cfg.seed = args.seed
    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs
    
    # 3. 性能优化
    cudnn.benchmark = True
    
    # 4. 运行器初始化和执行
    runner = Runner(cfg)
    if args.validate:
        runner.validate()
    else:
        runner.train()
```

## 配置集成

### 配置加载
使用 MMEngine 的 `Config.fromfile()` 加载基于 Python 的配置文件:

```python
cfg = Config.fromfile(args.config)
```

**支持的配置格式**:
- 包含配置字典的 Python 文件
- 分层配置继承
- 动态配置生成

### 配置覆盖
命令行参数覆盖配置文件值:

```python
cfg.gpus = len(args.gpus)          # GPU 数量
cfg.load_from = args.load_from     # 检查点路径
cfg.view = args.view               # 可视化标志
cfg.seed = args.seed               # 随机种子
cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs
```

这允许灵活的实验而不需要修改配置文件。

## GPU 管理

### 环境变量设置
```python
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpus)
```

**作用**: 控制 PyTorch 可见的 GPU
**优势**:
- 将训练隔离到特定的 GPU
- 在选定的设备上启用多 GPU 训练
- 防止与其他进程干扰

### 性能优化
```python
cudnn.benchmark = True
```

**作用**: 启用 cuDNN 自动调优器以获得最优卷积算法
**优势**:
- 自动选择最快的卷积实现
- 提高固定输入尺寸的训练速度
- 针对特定硬件配置进行优化

## 使用示例

### 训练示例

**单 GPU 训练**:
```bash
python tools/main.py configs/exp_srlane_culane.py --gpus 0
```

**多 GPU 训练**:
```bash
python tools/main.py configs/exp_srlane_culane.py --gpus 0 1 2 3
```

**自定义设置训练**:
```bash
python tools/main.py configs/exp_srlane_culane.py \
    --gpus 0 1 \
    --work_dirs ./custom_output \
    --seed 42
```

**从检查点恢复训练**:
```bash
python tools/main.py configs/exp_srlane_culane.py \
    --load_from ./work_dirs/ckpt/best_model.pth \
    --gpus 0
```

### 验证示例

**标准验证**:
```bash
python tools/main.py configs/exp_srlane_culane.py \
    --load_from checkpoint/baseline.pth \
    --validate \
    --gpus 0
```

**带可视化的验证**:
```bash
python tools/main.py configs/exp_srlane_culane.py \
    --load_from checkpoint/baseline.pth \
    --validate \
    --view \
    --gpus 0
```

## 模式选择

### 训练模式（默认）
- 当未提供 `--validate` 标志时执行
- 运行完整的训练流程并定期验证
- 保存检查点并记录训练进度
- 在指定间隔自动验证

### 验证模式
- 通过 `--validate` 标志激活
- 从 `--load_from` 指定的检查点加载模型
- 在验证集上运行推理
- 计算并报告评估指标
- 可选地使用 `--view` 标志可视化结果

## 错误处理和验证

### 必需参数
- 脚本验证是否提供了配置文件路径
- 如果缺少必需参数，则退出并显示使用消息

### 配置验证
- MMEngine 配置系统验证配置文件语法
- 报告配置问题的详细错误消息

### GPU 可用性
- 依赖 PyTorch CUDA 可用性检查
- 有条件地应用 cuDNN 基准测试优化

## 与 SRLane 架构的集成

### 组件协调
主脚本协调多个 SRLane 组件:

1. **配置系统**: 加载和合并实验设置
2. **Runner 类**: 将执行委托给训练/验证协调器
3. **GPU 管理**: 设置分布式训练环境
4. **日志记录**: 从 Runner 继承日志配置

### 实验管理
脚本支持各种实验工作流:

- **超参数扫描**: 使用相同脚本的不同配置
- **消融研究**: 修改特定配置组件
- **迁移学习**: 使用 `--load_from` 加载预训练权重
- **性能分析**: 仅验证运行用于模型评估

## 设计模式

1. **命令行接口**: 用于用户交互的标准 argparse 模式
2. **配置覆盖**: 命令行参数优先于配置文件
3. **模式选择**: 单个脚本处理训练和验证
4. **环境设置**: 系统化的 GPU 和性能配置
5. **委托**: 核心逻辑委托给专门的 Runner 类

主脚本为运行 SRLane 实验提供了一个清晰、灵活的接口，同时在简单的命令行接口后面处理 GPU 管理、配置加载和模式选择的复杂性。