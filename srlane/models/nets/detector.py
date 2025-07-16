# 这是SRLane项目中最重要的文件之一：两阶段检测器的定义
# 它实现了车道线检测的核心网络结构

import torch.nn as nn  # 导入PyTorch的神经网络模块

# 导入SRLane的注册系统和构建函数
from srlane.models.registry import NETS  # 网络注册器，用于注册不同的网络结构
from srlane.models.registry import build_backbones, build_head, build_neck  # 构建各个组件的工厂函数


@NETS.register_module  # 这个装饰器将这个类注册到系统中，以便在配置文件中使用
class TwoStageDetector(nn.Module):
    """两阶段检测器的基础类
    这是整个车道线检测系统的核心，它包括四个主要部分：
    1. backbone：骨干网络，负责提取图像特征
    2. neck：颈部网络，负责特征融合
    3. rpn_head：第一阶段检测头，生成车道线候选区域
    4. roi_head：第二阶段检测头，精化车道线检测结果

    Args:
        cfg: 模型配置对象，包含了所有的模型参数
    """
    def __init__(self, cfg):
        """初始化两阶段检测器"""
        super(TwoStageDetector, self).__init__()  # 调用父类的初始化方法
        self.cfg = cfg  # 保存配置对象
        
        # 构建四个主要组件：
        self.backbone = build_backbones(cfg)  # 骨干网络（通常是ResNet），用于提取图像特征
        self.neck = build_neck(cfg)  # 颈部网络（通常是FPN），用于融合不同层级的特征
        self.rpn_head = build_head(cfg.rpn_head, cfg)  # RPN头，第一阶段检测，生成车道线候选区域
        self.roi_head = build_head(cfg.roi_head, cfg)  # ROI头，第二阶段检测，精化车道线检测结果

    def extract_feat(self, batch):
        """提取图像特征的函数，这是整个检测流程的第一步"""
        # 从输入数据中获取图像，处理两种情况：
        # 1. 如果输入是字典，则取出"img"键对应的图像
        # 2. 如果输入直接是图像张量，则直接使用
        feat = self.backbone(batch["img"]
                             if isinstance(batch, dict) else batch)
        # 通过颈部网络融合不同层级的特征，得到最终的特征表示
        feat = self.neck(feat)
        return feat  # 返回提取到的特征

    def _forward_test(self, batch):
        """测试时的前向传播函数，只输出检测结果，不计算损失"""
        # 第一步：提取图像特征
        feat = self.extract_feat(batch)
        # 第二步：第一阶段检测，生成车道线候选区域
        rpn_result_dict = self.rpn_head(feat)
        # 第三步：第二阶段检测，精化车道线检测结果
        return self.roi_head(feat, **rpn_result_dict)

    def _forward_train(self, batch):
        """训练时的前向传播函数，需要计算损失函数用于反向传播更新参数"""
        # 第一步：提取图像特征
        feat = self.extract_feat(batch)
        # 初始化损失字典，用于存储各种损失值
        loss_dic = dict()

        # 第二步：第一阶段检测（RPN）
        rpn_result_dict = self.rpn_head(feat)  # 得到RPN的检测结果
        rpn_loss = self.rpn_head.loss(**rpn_result_dict, **batch)  # 计算RPN的损失
        loss_dic.update(rpn_loss)  # 将RPN损失加入总损失字典
        
        # 第三步：第二阶段检测（ROI）
        roi_result_dict = self.roi_head(feat, **rpn_result_dict)  # 得到ROI的检测结果
        roi_loss = self.roi_head.loss(roi_result_dict, batch=batch)  # 计算ROI的损失
        loss_dic.update(roi_loss)  # 将ROI损失加入总损失字典

        # 第四步：为不同的损失加权重（不同损失的重要性不同）
        for loss_k, loss_v in loss_dic.items():
            # 从配置文件中获取损失权重，如果没有设置则默认为1.0
            loss_dic[loss_k] = loss_v * self.cfg.get(f"{loss_k}_weight", 1.)
        
        # 第五步：计算总损失（所有损失的加权和）
        all_loss = sum(loss_dic.values())
        loss_dic["loss"] = all_loss  # 将总损失也加入字典

        # 返回损失信息，用于反向传播和日志记录
        return {"loss": all_loss,      # 用于反向传播的总损失
                "loss_status": loss_dic}  # 用于日志显示的详细损失信息

    def forward(self, batch):
        """主要的前向传播函数，根据模型状态选择训练或测试模式"""
        if self.training:  # 如果模型处于训练模式
            return self._forward_train(batch)  # 调用训练前向传播，返回损失信息
        return self._forward_test(batch)  # 否则调用测试前向传播，返回检测结果

    def __repr__(self):
        """返回模型的详细信息字符串，包括各个组件和参数数量"""
        # 分隔符，用于美化输出格式
        separator_info = "======== Param. Info. ========"
        # 计算模型总参数数量：遍历所有参数，统计元素个数
        num_params = sum(map(lambda x: x.numel(), self.parameters()))
        # 格式化参数信息字符串
        info = f"#Params of {self._get_name()}: "
        info += f"{num_params / 10 ** 6:<.2f}[M]"  # 转换为百万单位显示
        # 返回包含所有组件信息的字符串
        return '\n'.join([separator_info, repr(self.backbone), repr(self.neck),
                          repr(self.rpn_head), repr(self.roi_head), info])
