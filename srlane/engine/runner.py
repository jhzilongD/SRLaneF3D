# 这是SRLane的主要训练引擎文件，负责整个训练和验证流程
# 使用Lightning Fabric来加速训练过程，支持多卡训练

import time  # 时间测量工具
import random  # 随机数生成器

import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示工具
from lightning.fabric import Fabric  # Lightning的轻量级加速工具

from .optimizer import build_optimizer  # 优化器构建函数
from .scheduler import build_scheduler  # 学习率调度器构建函数
from srlane.models.registry import build_net  # 网络构建函数
from srlane.datasets import build_dataloader  # 数据加载器构建函数
from srlane.utils.recorder import build_recorder  # 训练记录器构建函数
from srlane.utils.net_utils import save_model, load_network  # 模型保存和加载工具


class Runner(object):
    """训练和验证的主要执行器类
    
    这个类封装了整个训练流程，包括：
    1. 模型、优化器、调度器的初始化
    2. 训练循环和验证流程
    3. 模型保存和加载
    4. 多卡训练支持
    """
    def __init__(self, cfg):
        # 设置随机种子，保证结果可重现
        torch.manual_seed(cfg.seed)  # PyTorch随机种子
        np.random.seed(cfg.seed)     # NumPy随机种子
        random.seed(cfg.seed)        # Python随机种子
        
        self.cfg = cfg  # 保存配置对象
        self.recorder = build_recorder(self.cfg)  # 创建训练记录器，记录损失、指标等
        self.net = build_net(cfg)  # 构建神经网络模型
        self.load_network()  # 加载预训练模型（如果指定了）
        self.optimizer = build_optimizer(self.cfg, self.net)  # 创建优化器
        self.scheduler = build_scheduler(self.cfg, self.optimizer)  # 创建学习率调度器
        
        # 初始化Lightning Fabric用于多卡训练
        self.fabric = Fabric(accelerator="cuda",    # 使用CUDA加速
                             devices=cfg.gpus,      # GPU数量
                             strategy="dp",         # 数据并行策略
                             precision=cfg.precision)  # 数值精度
        self.fabric.launch()  # 启动Fabric
        # 设置模型和优化器以适配多卡训练
        self.net, self.optimizer = self.fabric.setup(self.net, self.optimizer)

        # 初始化数据加载器和评估指标
        self.val_loader = None  # 验证数据加载器，延迟初始化
        self.test_loader = None  # 测试数据加载器，延迟初始化
        self.metric = 0  # 当前最佳评估指标

    def load_network(self):
        """加载预训练模型的函数"""
        if not self.cfg.load_from:  # 如果没有指定加载路径
            return  # 直接返回，不加载任何模型
        # 加载预训练模型，strict=False允许部分参数不匹配
        load_network(self.net, self.cfg.load_from, strict=False)

    def train_epoch(self, train_loader):
        """单个训练epoch的执行函数
        
        Args:
            train_loader: 训练数据加载器
        """
        self.net.train()  # 设置模型为训练模式
        end = time.time()  # 记录开始时间
        
        for i, data in enumerate(train_loader):  # 遍历训练数据
            if self.recorder.step >= self.cfg.total_iter:  # 达到最大迭代次数
                break  # 退出训练循环
                
            date_time = time.time() - end  # 计算数据加载时间
            self.recorder.step += 1  # 更新训练步数
            
            # 前向传播
            output = self.net(data)  # 通过网络获得输出结果
            
            # 反向传播和优化
            self.optimizer.zero_grad()  # 清零梯度
            loss = output["loss"].sum()  # 计算总损失
            self.fabric.backward(loss)  # 反向传播计算梯度
            self.optimizer.step()  # 更新模型参数
            self.scheduler.step()  # 更新学习率
            
            # 计算和记录时间
            batch_time = time.time() - end  # 计算批处理时间
            end = time.time()  # 重新记录时间
            
            # 更新记录器信息
            self.recorder.update_loss_status(output["loss_status"])  # 更新损失状态
            self.recorder.batch_time.update(batch_time)  # 更新批处理时间
            self.recorder.data_time.update(date_time)  # 更新数据加载时间

            # 定期记录训练信息
            if i % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]  # 获取当前学习率
                self.recorder.lr = lr  # 记录学习率
                self.recorder.record("train")  # 记录训练信息

    def train(self):
        """主训练函数，执行完整的训练流程"""
        self.recorder.logger.info("Build train_loader...")  # 日志信息
        # 构建训练数据加载器
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)
        
        # 打印数据集和数据加载器信息
        print("\n" + "="*60)
        print("CULane训练数据集信息：")
        print(f"数据集样本数量: {len(train_loader.dataset)} 张图像")
        print(f"批次大小 (batch_size): {train_loader.batch_size}")
        print(f"数据加载器批次数量: {len(train_loader)} 个批次")
        print(f"工作进程数 (num_workers): {train_loader.num_workers}")
        print(f"是否打乱数据 (shuffle): {train_loader._DataLoader__initialized and hasattr(train_loader.sampler, 'shuffle')}")
        print("="*60 + "\n")
        
        # 使用Fabric设置数据加载器以适配多卡训练
        train_loader = self.fabric.setup_dataloaders(train_loader)
        self.recorder.logger.info("Start training...")  # 输出开始训练信息
        
        epoch = 0  # 初始化epoch计数器
        # 训练主循环，直到达到最大迭代次数
        while self.recorder.step < self.cfg.total_iter:
            self.recorder.epoch = epoch  # 记录当前epoch
            self.train_epoch(train_loader)  # 执行一个epoch的训练
            
            # 检查是否需要进行验证
            if (self.recorder.step >= self.cfg.total_iter  # 达到最大迭代次数
                    or (epoch + 1) % self.cfg.eval_ep == 0):  # 或达到验证间隔
                self.validate()  # 执行验证
            epoch += 1  # 更新epoch计数器

    @torch.no_grad()  # 验证阶段不需要计算梯度
    def validate(self):
        """模型验证函数，评估模型在验证集上的性能"""
        # 延迟初始化验证数据加载器
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)  # 验证模式
        
        net = self.net  # 获取网络模型
        net.eval()  # 设置模型为评估模式
        predictions = []  # 存储所有预测结果
        
        # 遍历验证数据集
        for i, data in enumerate(tqdm(self.val_loader, desc="Validate")):
            output = net(data)  # 模型前向传播
            # 通过ROI头获取最终的车道线结果
            output = net.module.roi_head.get_lanes(output, data["meta"])
            predictions.extend(output)  # 添加到预测结果列表
            
            # 如果需要可视化结果
            if self.cfg.view:
                self.val_loader.dataset.view(output, data["meta"])

        # 使用数据集的评估函数计算指标
        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info("metric: " + str(metric))  # 记录指标信息
        # 将指标记录到TensorBoard
        self.recorder.tb_logger.add_scalar("val/metric", metric,
                                           self.recorder.step)
        
        # 如果当前指标超过历史最佳值，保存模型
        if metric > self.metric:
            self.metric = metric  # 更新最佳指标
            save_model(net, self.recorder)  # 保存模型
