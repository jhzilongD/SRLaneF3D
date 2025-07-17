# 继承基础配置文件，MMEngine会自动合并这些配置
_base_ = [
    "./datasets/culane.py",    # CULane数据集相关配置（数据路径、预处理、数据增强等）
    "./models/srlane_r18.py"    # SRLane模型配置（使用ResNet-18作为backbone）
]

# 训练输出目录，包含模型权重、日志、TensorBoard文件等
work_dirs = "work_dirs/sr_cu"

# 损失函数权重配置（用于多任务学习的损失平衡）
iou_loss_weight = 2.        # LineIOU损失权重，用于优化车道线与真值的重叠度
cls_loss_weight = 2.        # 分类损失权重，用于判断车道线的存在性
l1_loss_weight = 0.2        # L1回归损失权重，用于车道线点位置的精确回归
angle_loss_weight = 5       # 角度损失权重，用于第一阶段的局部角度预测
attn_loss_weight = 0.  # 0.05  # 注意力损失权重（当前未使用，可选的辅助监督）
seg_loss_weight = 0.  # 0.5    # 分割损失权重（当前未使用，可选的辅助监督）

# 训练参数配置
total_iter = 11110          # 总训练迭代次数
batch_size = 40             # 批量大小（每次迭代处理的图像数量）
eval_ep = 3                 # 验证频率，每3个epoch进行一次验证
workers = 8                 # 数据加载的并行进程数
log_interval = 500          # 日志打印间隔，每500次迭代打印一次训练信息

# 混合精度训练配置，使用FP16加速训练并减少显存占用
precision = "16-mixed"  # "32"  # 可选"32"使用FP32全精度训练

# 优化器配置，使用AdamW优化器
optimizer = dict(type="AdamW", lr=6e-4)  # 学习率设置为6e-4

# 学习率调度器配置，使用warmup策略
scheduler = dict(type="warmup", warm_up_iters=800, total_iters=total_iter)  # 前800次迭代进行warmup
