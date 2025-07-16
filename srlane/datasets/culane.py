# 这是CULane数据集的实现文件
# CULane是一个大型车道线检测数据集，包含了各种道路场景和天气条件

import os  # 操作系统接口
import os.path as osp  # 路径操作工具
from os.path import join  # 路径连接函数

import numpy as np  # 数值计算库
import pickle as pkl  # 序列化工具，用于缓存数据
from tqdm import tqdm  # 进度条显示工具

import srlane.evaluation.culane_metric as culane_metric  # CULane数据集的评估指标
from .base_dataset import BaseDataset  # 基础数据集类
from .registry import DATASETS  # 数据集注册器

# 定义不同划分集对应的文件列表
LIST_FILE = {
    "train": "list/train_gt.txt",  # 训练集文件列表
    "val": "list/val.txt",        # 验证集文件列表
    "test": "list/test.txt",      # 测试集文件列表
}

# 定义测试集中的不同场景类别，用于分类评估
CATEGORYS = {
    "normal": "list/test_split/test0_normal.txt",  # 正常场景
    "crowd": "list/test_split/test1_crowd.txt",    # 拥挤场景
    "hlight": "list/test_split/test2_hlight.txt",  # 高亮度场景
    "shadow": "list/test_split/test3_shadow.txt",  # 阴影场景
    "noline": "list/test_split/test4_noline.txt",  # 无车道线场景
    "arrow": "list/test_split/test5_arrow.txt",    # 带箭头的车道线
    "curve": "list/test_split/test6_curve.txt",    # 弯曲车道线
    "cross": "list/test_split/test7_cross.txt",    # 交叉口场景
    "night": "list/test_split/test8_night.txt",    # 夜间场景
}


@DATASETS.register_module  # 将这个类注册为数据集
class CULane(BaseDataset):
    """
CULane数据集类
    
    CULane是一个大型车道线检测数据集，包含：
    - 训练集：88,880张图像
    - 验证集：9,675张图像
    - 测试集：34,680张图像
    - 支持多种场景的分类评估
    """
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = join(data_root, LIST_FILE[split])  # 文件列表路径
        self.split = split  # 数据集划分（train/val/test）
        self.load_annotations()  # 加载注释数据
        # 定义采样点的y坐标，用于评估，从270到590像素，间隔8像素
        self.h_samples = np.arange(270, 590, 8) / 590  # 归一化到[0,1]范围

    def load_annotations(self, diff_thr=15):
        """加载注释数据的函数
        
        Args:
            diff_thr: 图像差异阈值，用于过滤相似图像
        """
        self.logger.info("Loading CULane annotations...")  # 日志信息
        os.makedirs(".cache", exist_ok=True)  # 创建缓存目录
        cache_path = f".cache/culane_{self.split}.pkl"  # 缓存文件路径
        
        # 如果缓存文件存在，直接加载
        if osp.exists(cache_path):
            with open(cache_path, "rb") as cache_file:
                self.data_infos = pkl.load(cache_file)  # 加载缓存数据
                # 计算最大车道数
                self.max_lanes = max(
                    len(anno["lanes"]) for anno in self.data_infos)
                return

        # 如果缓存不存在，重新加载注释
        self.data_infos = []
        with open(self.list_path) as list_file:
            prev_img = np.zeros(1)  # 上一张图像，用于计算差异
            for i, line in tqdm(enumerate(list_file)):  # 使用进度条显示进度
                infos = {}  # 存储单个样本的信息
                line = line.split()  # 分割行内容
                img_line = line[0]  # 图像路径
                # 处理路径格式，去除开头的'/'
                img_line = img_line[1 if img_line[0] == '/' else 0::]
                img_path = join(self.data_root, img_line)  # 完整图像路径
                
                # 在训练集中过滤相似图像，减少冗余数据
                if self.split == "train":
                    img = self.imread(img_path)  # 读取图像
                    # 计算与上一张图像的差异
                    diff = np.abs(img.astype(np.float32) -
                                  prev_img.astype(np.float32)).sum()
                    diff /= (img.shape[0] * img.shape[1] * img.shape[2])  # 平均差异
                    prev_img = img  # 更新上一张图像
                    if diff < diff_thr:  # 如果差异太小，跳过这张图像
                        continue
                        
                infos["img_name"] = img_line  # 图像名称
                infos["img_path"] = img_path  # 图像路径

                # 处理掩码路径（如果存在）
                if len(line) > 1:
                    mask_line = line[1]
                    mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
                    mask_path = join(self.data_root, mask_line)
                    infos["mask_path"] = mask_path

                # 处理车道线存在标记（如果存在）
                if len(line) > 2:
                    exist_list = [int(marker) for marker in line[2:]]
                    infos["lane_exist"] = np.array(exist_list)

                # 加载车道线注释文件
                anno_path = img_path[:-3] + "lines.txt"  # 将.jpg换成lines.txt
                with open(anno_path, 'r') as anno_file:
                    # 读取注释数据，每行包含一条车道线的坐标点
                    data = [
                        list(map(float, line.split()))
                        for line in anno_file.readlines()
                    ]
                    
                # 将数据转换为坐标点对的列表
                lanes = [
                    [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                     if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
                lanes = [list(set(lane)) for lane in
                         lanes]  # 移除重复点
                lanes = [lane for lane in lanes
                         if
                         len(lane) > 2]  # 移除少于2个点的车道线

                # 按y坐标排序，保证车道线点有序
                lanes = [sorted(lane, key=lambda x: x[1])
                         for lane in lanes]
                infos["lanes"] = lanes  # 保存车道线数据
                self.data_infos.append(infos)  # 添加到数据列表

        # 保存到缓存文件
        with open(cache_path, "wb") as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def get_prediction_string(self, pred):
        """将预测结果转换为CULane格式的字符串
        
        Args:
            pred: 预测的车道线列表
            
        Returns:
            CULane格式的预测结果字符串
        """
        ys = self.h_samples  # 获取采样点的y坐标
        out = []  # 存储输出结果
        
        for lane in pred:  # 遍历每条车道线
            xs = lane(ys)  # 根据y坐标计算对应的x坐标
            valid_mask = (xs >= 0) & (xs < 1)  # 找到有效的点（在图像范围内）
            # 将归一化坐标转换回像素坐标
            lane_xs = xs[valid_mask] * self.cfg.ori_img_w
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            # 反向排序，使得点从上到下排列
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            # 将坐标点转换为字符串格式
            lane_str = ' '.join([
                f"{x:.5f} {y:.5f}" for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':  # 如果字符串不为空
                out.append(lane_str)  # 添加到输出列表

        return '\n'.join(out)  # 返回用换行符连接的字符串

    def evaluate(self, predictions, output_basedir):
        """评估预测结果的函数
        
        Args:
            predictions: 模型的预测结果列表
            output_basedir: 输出结果的基本目录
            
        Returns:
            F1分数
        """
        self.logger.info("Generating CULane prediction output...")  # 日志信息
        
        # 遍历每个预测结果，生成输出文件
        for idx, pred in enumerate(predictions):
            # 构建输出目录路径
            output_dir = join(
                output_basedir,
                osp.dirname(self.data_infos[idx]["img_name"]))
            # 构建输出文件名，将.jpg换成lines.txt
            output_filename = osp.basename(
                self.data_infos[idx]["img_name"])[:-3] + "lines.txt"
            os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
            output = self.get_prediction_string(pred)  # 获取预测结果字符串

            # 将预测结果写入文件
            with open(join(output_dir, output_filename),
                      'w') as out_file:
                out_file.write(output)
                
        # 如果是测试集，对每个场景类别进行分别评估
        if self.split == "test":
            for cate, cate_file in CATEGORYS.items():
                culane_metric.eval_predictions(output_basedir,
                                               self.data_root,
                                               join(self.data_root, cate_file),
                                               iou_thresholds=[0.5],  # IoU阈值
                                               official=True)  # 使用官方评估方法

        # 对整个数据集进行评估
        result = culane_metric.eval_predictions(output_basedir,
                                                self.data_root,
                                                self.list_path,
                                                iou_thresholds=[0.5],
                                                official=True)

        return result[0.5]["F1"]  # 返回F1分数
