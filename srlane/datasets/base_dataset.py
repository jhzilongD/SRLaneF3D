# 这是SRLane数据集的基类，定义了所有数据集的通用功能
# 包括数据加载、预处理、可视化等核心功能

import logging  # 用于日志记录
import os.path as osp  # 路径操作工具

import cv2  # OpenCV图像处理库
from torch.utils.data import Dataset  # PyTorch数据集基类
from mmcv.parallel import DataContainer as DC  # MMEngine的数据容器

from .registry import DATASETS  # 数据集注册器
from .process import Process  # 数据预处理模块
from srlane.utils.visualization import imshow_lanes  # 车道线可视化工具


@DATASETS.register_module  # 将此类注册为数据集
class BaseDataset(Dataset):
    """车道线检测数据集的基类
    
    这个基类提供了所有车道线检测数据集的通用功能，包括：
    1. 数据加载和预处理
    2. 图像裁剪处理
    3. 车道线坐标转换
    4. 可视化功能
    """
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg  # 保存配置对象
        self.logger = logging.getLogger(__name__)  # 创建日志记录器
        self.data_root = data_root  # 数据集根目录
        self.training = "train" in split  # 判断是否为训练模式
        self.processes = Process(processes, cfg)  # 创建数据预处理流水线

    def view(self, predictions, img_metas):
        """可视化预测结果的函数
        
        这个函数用于在验证或测试阶段可视化模型的预测结果
        """
        # 展平元数据列表，因为DataContainer可能包含嵌套结构
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        
        # 遍历每个预测结果和对应的图像元数据
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta["img_name"]  # 获取图像名称
            # 读取原始图像
            img = cv2.imread(osp.join(self.data_root, img_name))
            # 构建输出文件路径，将路径中的'/'替换为'_'避免路径问题
            out_file = osp.join(self.cfg.work_dir, "visualization",
                                img_name.replace('/', '_'))
            # 将车道线对象转换为数组格式，适配原始图像尺寸
            lanes = [lane.to_array(img_meta["img_size"]) for lane in lanes]
            # 调用可视化函数保存结果
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data_infos)  # 返回数据信息列表的长度

    @staticmethod
    def imread(path, rgb=True):
        """静态方法：读取图像文件
        
        Args:
            path: 图像文件路径
            rgb: 是否转换为RGB格式（默认True）
        
        Returns:
            读取的图像数组
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # 读取彩色图像
        if rgb:  # 如果需要RGB格式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 从BGR转换为RGB
        return img

    def __getitem__(self, idx):
        """获取数据集中的一个样本
        
        这是PyTorch数据集的核心方法，负责加载和预处理单个样本
        """
        data_info = self.data_infos[idx]  # 获取指定索引的数据信息
        img = self.imread(data_info["img_path"])  # 读取图像
        
        # 裁剪图像：去除上方的天空部分，只保留道路部分
        img = img[self.cfg.cut_height:, :, :]
        
        sample = data_info.copy()  # 复制数据信息
        sample.update({"img": img})  # 更新图像数据

        # 如果是训练模式且进行了图像裁剪，需要调整车道线坐标
        if self.training:
            if self.cfg.cut_height != 0:
                new_lanes = []  # 存储调整后的车道线坐标
                for i in sample["lanes"]:  # 遍历每条车道线
                    lanes = []  # 存储单条车道线的坐标点
                    for p in i:  # 遍历车道线上的每个点
                        # 调整y坐标：减去裁剪的高度
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({"lanes": new_lanes})  # 更新车道线坐标

        # 通过预处理流水线处理样本
        sample = self.processes(sample)
        
        # 构建元数据信息，用于后续处理和可视化
        meta = {"full_img_path": data_info["img_path"],  # 完整图像路径
                "img_name": data_info["img_name"],          # 图像名称
                "img_size": data_info.get("img_size",       # 图像尺寸
                                          (self.cfg.ori_img_h,
                                           self.cfg.ori_img_w)),
                "img_cut_height": self.cfg.cut_height}      # 裁剪高度
        meta = DC(meta, cpu_only=True)  # 包装为数据容器，只在CPU上处理
        sample.update({"meta": meta})  # 添加元数据到样本

        return sample  # 返回处理后的样本
