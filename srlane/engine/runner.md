# 训练运行器 (`runner.py`)

## 文件概述

Runner 类是 SRLane 训练和验证流程的中央协调器。它集成了 PyTorch Lightning Fabric 进行分布式训练，管理完整的训练生命周期，并协调模型、数据、优化和评估组件之间的关系。

## 类: `Runner`

### 初始化 (`__init__`)

**作用**: 设置完整的训练环境，包括模型、优化、分布式训练和日志记录。

**参数**:
- `cfg`: 包含所有训练参数的配置对象

**关键初始化步骤**:

```python
def __init__(self, cfg):
    # 1. 可重现性设置
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    # 2. 核心组件
    self.cfg = cfg
    self.recorder = build_recorder(self.cfg)  # 日志和指标
    self.net = build_net(cfg)                # 模型架构
    self.load_network()                      # 加载预训练权重
    self.optimizer = build_optimizer(self.cfg, self.net)  # 优化器
    self.scheduler = build_scheduler(self.cfg, self.optimizer)  # 学习率调度器
    
    # 3. 分布式训练设置
    self.fabric = Fabric(
        accelerator="cuda",
        devices=cfg.gpus,
        strategy="dp",          # 数据并行策略
        precision=cfg.precision
    )
    self.fabric.launch()
    self.net, self.optimizer = self.fabric.setup(self.net, self.optimizer)
    
    # 4. 状态变量
    self.val_loader = None
    self.test_loader = None
    self.metric = 0  # 最佳验证指标
```

**组件集成**:
- **Recorder**: 处理日志、指标跟踪和 tensorboard 集成
- **Network**: 车道线检测模型 (TwoStageDetector)
- **Optimizer**: 参数优化 (Adam, SGD 等)
- **Scheduler**: 学习率调度
- **Fabric**: PyTorch Lightning 分布式训练包装器

### 网络加载 (`load_network`)

**作用**: 如果在配置中指定，则加载预训练权重。

```python
def load_network(self):
    if not self.cfg.load_from:
        return
    load_network(self.net, self.cfg.load_from, strict=False)
```

**特性**:
- **条件加载**: 仅在指定路径时加载
- **非严格加载**: 允许为迁移学习进行部分权重加载
- **检查点恢复**: 支持从训练检查点恢复

### 训练方法

#### `train_epoch(train_loader)`

**作用**: 执行一个完整的训练轮次，包括梯度更新和日志记录。

**参数**:
- `train_loader`: 用于训练数据的 DataLoader

**训练循环数据流**:

```python
def train_epoch(self, train_loader):
    self.net.train()  # 设置为训练模式
    end = time.time()
    
    for i, data in enumerate(train_loader):
        # 提早停止条件
        if self.recorder.step >= self.cfg.total_iter:
            break
            
        # 计时和步骤跟踪
        date_time = time.time() - end
        self.recorder.step += 1
        
        # 前向传播
        output = self.net(data)  # 返回损失和预测
        
        # 反向传播
        self.optimizer.zero_grad()
        loss = output["loss"].sum()  # 聚合损失
        self.fabric.backward(loss)   # 分布式反向传播
        self.optimizer.step()
        self.scheduler.step()
        
        # 计时和日志
        batch_time = time.time() - end
        end = time.time()
        
        # 更新指标
        self.recorder.update_loss_status(output["loss_status"])
        self.recorder.batch_time.update(batch_time)
        self.recorder.data_time.update(date_time)
        
        # 定期日志记录
        if i % self.cfg.log_interval == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            self.recorder.lr = lr
            self.recorder.record("train")
```

**关键特性**:
- **基于迭代的训练**: 使用总迭代次数而不是轮次
- **分布式反向传播**: 利用 Fabric 进行梯度同步
- **全面计时**: 跟踪数据加载和计算时间
- **损失聚合**: 处理来自两阶段检测器的多组件损失
- **实时日志**: 定期记录指标和学习率

#### `train()`

**作用**: 协调轮次和验证的主训练循环。

**训练流程**:

```python
def train(self):
    # 设置数据加载
    self.recorder.logger.info("Build train_loader...")
    train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)
    train_loader = self.fabric.setup_dataloaders(train_loader)
    
    # 主训练循环
    self.recorder.logger.info("Start training...")
    epoch = 0
    while self.recorder.step < self.cfg.total_iter:
        self.recorder.epoch = epoch
        self.train_epoch(train_loader)
        
        # 验证条件
        if (self.recorder.step >= self.cfg.total_iter or 
            (epoch + 1) % self.cfg.eval_ep == 0):
            self.validate()
        epoch += 1
```

**控制流**:
1. **数据加载器设置**: 创建和配置训练数据加载器
2. **Fabric 集成**: 为分布式训练包装数据加载器
3. **轮次循环**: 继续直到达到迭代限制
4. **验证触发**: 在指定间隔或完成时进行验证

### 验证方法

#### `validate()`

**作用**: 在验证集上执行模型评估并计算指标。

**验证流程**:

```python
@torch.no_grad()
def validate(self):
    # 延迟数据加载器初始化
    if not self.val_loader:
        self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
    
    net = self.net
    net.eval()  # 设置为评估模式
    predictions = []
    
    # 推理循环
    for i, data in enumerate(tqdm(self.val_loader, desc="Validate")):
        output = net(data)  # 前向传播
        # 从 ROI 头提取车道线预测
        output = net.module.roi_head.get_lanes(output, data["meta"])
        predictions.extend(output)
        
        # 可选可视化
        if self.cfg.view:
            self.val_loader.dataset.view(output, data["meta"])
    
    # 评估和日志记录
    metric = self.val_loader.dataset.evaluate(predictions, self.cfg.work_dir)
    self.recorder.logger.info("metric: " + str(metric))
    self.recorder.tb_logger.add_scalar("val/metric", metric, self.recorder.step)
    
    # 最佳模型保存
    if metric > self.metric:
        self.metric = metric
        save_model(net, self.recorder)
```

**关键特性**:
- **无梯度计算**: 使用 `@torch.no_grad()` 装饰器提高效率
- **延迟数据加载**: 仅在需要时创建验证加载器
- **车道线提取**: 处理模型输出以提取车道线预测
- **可视化支持**: 验证期间可选的结果可视化
- **指标跟踪**: 使用数据集特定指标进行全面评估
- **最佳模型保存**: 自动保存性能改进的模型

## 数据流和张量形状

### 训练数据流
1. **输入数据**: 带有元数据的 `[B, 3, H, W]` 图像张量
2. **模型前向**: 产生损失字典和中间输出
3. **损失聚合**: 汇总来自不同检测头的组件损失
4. **梯度计算**: 通过 Fabric 进行分布式反向传播
5. **参数更新**: 优化器步骤并调整调度器

### 验证数据流
1. **输入数据**: `[B, 3, H, W]` 验证图像
2. **模型推理**: 生成原始预测
3. **后处理**: ROI 头提取车道线坐标
4. **评估**: 数据集特定的指标计算
5. **日志记录**: TensorBoard 和控制台输出

## 与 SRLane 架构的集成

### 组件依赖
- **模型**: 使用 `build_net()` 实例化 TwoStageDetector
- **数据集**: 与 CULane/TuSimple 数据集实现集成
- **优化**: 通过工厂函数协调优化器和调度器
- **日志记录**: 通过记录器系统进行全面的指标跟踪

### 配置集成
```python
# 关键配置参数
cfg.total_iter = 80000      # 训练迭代次数
cfg.eval_ep = 5             # 验证频率（轮次）
cfg.log_interval = 50       # 日志频率（迭代）
cfg.gpus = [0, 1]          # GPU 设备
cfg.precision = "16-mixed"  # 混合精度训练
cfg.seed = 0               # 可重现性种子
```

### 分布式训练特性
- **多 GPU 支持**: 跨多个 GPU 的数据并行训练
- **混合精度**: 自动混合精度以提高内存效率
- **梯度同步**: 跨设备的自动梯度平均化
- **数据加载**: 分布式数据采样和加载

Runner 类作为中央协调器，统筹 SRLane 训练的所有方面，从数据加载和模型前向传播到优化和验证，同时提供全面的日志记录和分布式训练能力。