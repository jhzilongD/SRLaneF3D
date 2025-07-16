# 这是SRLane中的局部角度头文件，实现了第一阶段的车道线候选区域生成
# 通过预测角度图来生成车道线的初始候选区域，是两阶段检测的第一步

import math  # 用于数学计算，特别是角度相关的计算
from typing import Tuple, List  # 类型提示

import torch  # PyTorch核心库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数库
from torch import Tensor  # 张量类型

from srlane.models.registry import HEADS  # 检测头注册器
from srlane.models.losses.seg_loss import SegLoss  # 分割损失函数


@HEADS.register_module  # 将这个类注册为检测头
class LocalAngleHead(nn.Module):
    """局部角度预测头（第一阶段检测头）
    
    这个模块是SRLane两阶段检测系统的第一阶段，负责：
    1. 预测每个像素点的局部角度信息
    2. 生成车道线的初始候选区域
    3. 可选地进行语义分割辅助训练
    
    Args:
        num_points: 车道线点的数量，默认72个点
        in_channel: 输入特征的通道数
        cfg: 模型配置对象
    """

    def __init__(self,
                 num_points: int = 72,    # 车道线点的数量
                 in_channel: int = 64,    # 输入特征的通道数
                 cfg=None,                # 模型配置对象
                 ):
        super(LocalAngleHead, self).__init__()
        self.n_offsets = num_points  # 车道线点的数量
        self.cfg = cfg  # 保存配置对象
        self.img_w = cfg.img_w  # 图像宽度
        self.img_h = cfg.img_h  # 图像高度
        # 是否使用辅助分割损失，用于提高训练效果
        self.aux_seg = self.cfg.get("seg_loss_weight", 0.) > 0.
        self.feat_h, self.feat_w = self.cfg.angle_map_size  # 特征图的尺寸
        
        # 笛卡尔坐标系统中的先验信息
        # 创建从上到下的y坐标序列，用于车道线采样
        self.register_buffer(name="prior_ys",
                             tensor=torch.linspace(0, self.feat_h,
                                                   steps=self.n_offsets,
                                                   dtype=torch.float32))
        
        # 创建坐标网格，用于角度到坐标的转换
        # grid_y: 从上到下的y坐标
        grid_y, grid_x = torch.meshgrid(torch.arange(self.feat_h - 0.5, 0,
                                                     -1, dtype=torch.float32),
                                        # grid_x: 从左到右的x坐标
                                        torch.arange(0.5, self.feat_w,
                                                     1, dtype=torch.float32),
                                        indexing="ij")
        grid = torch.stack((grid_x, grid_y), 0)  # 将x和y坐标合并
        grid.unsqueeze_(0)  # 扩展维度为 (1, 2, h, w)
        self.register_buffer(name="grid", tensor=grid)  # 注册为缓冲区

        # 创建角度预测卷积层列表，对应特征金字塔的每一层
        self.angle_conv = nn.ModuleList()
        for _ in range(self.cfg.n_fpn):  # 遵循特征金字塔的层数
            # 1x1卷积层，将多通道特征减少为单通道角度预测
            self.angle_conv.append(nn.Conv2d(in_channel, 1,  # 输出1个通道
                                             1, 1, 0, bias=False))  # 1x1卷积，无偏置

        # 如果需要辅助分割损失
        if self.aux_seg:
            num_classes = self.cfg.max_lanes + 1  # 分割类别数：最大车道数+1（背景）
            self.seg_conv = nn.ModuleList()  # 分割卷积层列表
            for _ in range(self.cfg.n_fpn):
                # 分割卷积层，输出多个类别的分割结果
                self.seg_conv.append(nn.Conv2d(in_channel, num_classes,
                                               1, 1, 0))  # 1x1卷积
            self.seg_criterion = SegLoss(num_classes=num_classes)  # 分割损失函数
        self.init_weights()  # 初始化网络权重

    def init_weights(self):
        """初始化网络权重的函数"""
        # 为角度预测层初始化权重，使用较小的正态分布
        for m in self.angle_conv.parameters():
            nn.init.normal_(m, 0., 1e-3)  # 均值0，标准差0.001的正态分布

    def forward(self,
                feats: List[Tensor], ):
        """执行局部角度头的前向传播过程

        Args:
        - feats: 特征金字塔的特征图列表，包含多个尺度的特征

        Returns:
        - dict: 包含车道线候选区域和预测结果的字典
        """
        theta_list = []  # 存储每个尺度的角度预测结果
        
        # 在测试模式下，只使用最深层的特征来提高推理速度
        if not self.training:
            feats = feats[-1:]  # 只保留最后一个特征图
        
        # 对每个尺度的特征进行角度预测
        for i, feat in enumerate(feats, 1):
            # 通过对应的卷积层预测角度，并使用sigmoid确保输出在[0,1]范围内
            theta = self.angle_conv[len(feats) - i](feat).sigmoid()
            theta_list.append(theta)  # 添加到角度列表
            
        # 如果需要辅助分割损失，同时计算分割结果
        if self.aux_seg:
            seg_list = []  # 存储每个尺度的分割预测结果
            for i, feat in enumerate(feats, 1):
                # 通过对应的分割卷积层预测分割结果
                seg = self.seg_conv[len(feats) - i](feat)
                seg_list.append(seg)
                
        # 将最深层的角度预测上采样到指定尺寸
        angle = F.interpolate(theta_list[-1],
                              size=[self.feat_h, self.feat_w],  # 目标尺寸
                              mode="bilinear",      # 双线性插值
                              align_corners=True).squeeze(1)  # 去除通道维度
        angle = angle.detach()  # 停止梯度传播，避免影响后续计算
        
        # 移除过于倾斜的角度，可选操作
        angle.clamp_(min=0.05, max=0.95)  # 将角度约束在合理范围内
        
        # 构建车道线候选区域
        k = (angle * math.pi).tan()  # 将角度转换为斜率（正切值）
        bs, h, w = angle.shape  # 批大小、高度、宽度
        grid = self.grid  # 获取坐标网格
        
        # 根据角度和坐标网格计算车道线在每个采样点的x坐标
        ws = ((self.prior_ys.view(1, 1, self.n_offsets)  # 预定义的y坐标
               - grid[:, 1].view(1, h * w, 1)) / k.view(bs, h * w, 1)  # 根据斜率计算
              + grid[:, 0].view(1, h * w, 1))  # 加上起始点的x坐标
        ws = ws / w  # 归一化到[0,1]范围
        
        # 判断哪些点是有效的（在图像范围内）
        valid_mask = (0 <= ws) & (ws < 1)
        _, indices = valid_mask.max(-1)  # 找到每个候选区域的起始点
        start_y = indices / (self.n_offsets - 1)  # 计算起始点的y坐标
        
        # 初始化候选区域张量
        priors = ws.new_zeros(
            (bs, h * w, 2 + 2 + self.n_offsets), device=ws.device)
        priors[..., 2] = start_y  # 设置起始点的y坐标
        priors[..., 4:] = ws  # 设置车道线在每个采样点的x坐标

        return dict(priors=priors,  # 返回候选区域
                    pred_angle=[theta.squeeze(1) for theta in theta_list]
                    if self.training else None,  # 训练时返回角度预测
                    pred_seg=seg_list
                    if (self.training and self.aux_seg) else None)  # 训练时返回分割预测

    def loss(self,
             pred_angle: List[Tensor],      # 预测的角度图列表
             pred_seg: List[Tensor],        # 预测的分割图列表
             gt_angle: List[Tensor],        # 真实的角度图列表
             gt_seg: List[Tensor],          # 真实的分割图列表
             loss_weight: Tuple[float] = [0.2, 0.2, 1.],  # 各层损失的权重
             ignore_value: float = 0.,      # 忽略值，用于过滤非目标区域
             **ignore_kwargs):
        """在多层特征上计算局部角度估计的L1损失

        Args:
        - pred_angle: 预测的角度图列表
        - gt_angle: 真实的角度图列表
        - loss_weight: 每个层级损失的权重
        - ignore_value: 非目标区域的占位值

        Returns:
        - dict: 包含计算得到的损失值的字典
        """
        angle_loss = 0  # 初始化角度损失
        
        # 对每个层级计算角度损失
        for pred, target, weight in zip(pred_angle, gt_angle, loss_weight):
            valid_mask = target > ignore_value  # 创建有效区域的掩码
            # 计算加权的L1损失，只在有效区域计算
            angle_loss = (angle_loss
                          + ((pred - target).abs() * valid_mask).sum()  # 绝对值损失
                          / (valid_mask.sum() + 1e-4)) * weight  # 平均化并加权
        
        # 如果使用辅助分割损失
        if self.aux_seg:
            seg_loss = 0  # 初始化分割损失
            # 对每个层级计算分割损失
            for pred, target, weight in zip(pred_seg, gt_seg, loss_weight):
                seg_loss = seg_loss + self.seg_criterion(pred, target) * weight
            return {"angle_loss": angle_loss,  # 返回角度损失
                    "seg_loss": seg_loss, }   # 和分割损失

        return {"angle_loss": angle_loss}  # 只返回角度损失

    def __repr__(self):
        """返回模型的参数信息字符串"""
        num_params = sum(map(lambda x: x.numel(), self.parameters()))  # 计算总参数数量
        return f"#Params of {self._get_name()}: {num_params / 10 ** 3:<.2f}[K]"  # 返回参数信息
