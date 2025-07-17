import torch.nn as nn  # 导入PyTorch的神经网络模块
from srlane.registry import Registry, build_from_cfg  # 导入注册器和构建函数

BACKBONES = Registry("backbones")  # 创建backbone（主干网络）注册器实例
HEADS = Registry("heads")  # 创建head（检测头）注册器实例
NECKS = Registry("necks")  # 创建neck（颈部网络）注册器实例
NETS = Registry("nets")  # 创建net（网络）注册器实例


def build(cfg, registry, default_args=None):  # 通用构建函数，根据配置构建模块
    if isinstance(cfg, list):  # 如果配置是列表，说明需要构建多个模块
        print(f"cfg is list: {cfg}")  # 调试输出：配置是列表
        modules = [  # 创建模块列表
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg  # 遍历配置列表，构建每个模块
        ]
        return nn.Sequential(*modules)  # 将多个模块组合成顺序执行的Sequential模块
    else:  # 如果配置不是列表，说明只需要构建单个模块
        print(f"cfg is not list: {cfg}")  # 调试输出：配置不是列表
        return build_from_cfg(cfg, registry, default_args)  # 直接构建单个模块


def build_backbones(cfg):  # 构建主干网络的函数
    return build(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))  # 调用build函数，传入backbone配置和BACKBONES注册器


def build_neck(cfg):  # 构建颈部网络的函数
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))  # 调用build函数，传入neck配置和NECKS注册器


def build_head(split_cfg, cfg):  # 构建检测头的函数
    return build(split_cfg, HEADS, default_args=dict(cfg=cfg))  # 调用build函数，传入检测头配置和HEADS注册器


def build_net(cfg):  # 构建完整网络的函数
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))  # 调用build函数，传入网络配置和NETS注册器
