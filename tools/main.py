# 这是SRLane项目的主入口文件，负责启动整个车道线检测的训练或测试过程

import os  # 导入操作系统接口模块，用于设置环境变量等系统操作
import argparse  # 导入命令行参数解析模块，让我们可以在终端输入不同的参数来控制程序

import torch.backends.cudnn as cudnn  # PyTorch的CUDA深度神经网络库，用于加速GPU计算
from mmengine.config import Config  # MMEngine是一个深度学习框架，Config用于管理配置文件

from srlane.engine.runner import Runner  # 导入SRLane项目的核心运行器，它负责管理整个训练/测试流程


def parse_args():
    """解析命令行参数，这个函数让我们可以在终端输入不同的参数来控制程序行为"""
    # 创建一个命令行参数解析器，这就像是一个“输入解释器”
    parser = argparse.ArgumentParser(description="Train a detector")
    
    # 这是必需参数：配置文件路径（就像是一个“设置文件”，里面记录了所有训练参数）
    parser.add_argument("config",
                        help="Config file path")
    
    # 以下是可选参数：
    # 工作目录：保存训练日志和模型文件的地方
    parser.add_argument("--work_dirs", type=str, default=None,
                        help="Dirs for log and saving ckpts")
    
    # 加载检查点：如果你有一个已经训练好的模型，可以继续训练或者用来测试
    parser.add_argument("--load_from", default=None,
                        help="The checkpoint file to load from")
    
    # 可视化选项：在测试时是否显示检测结果的图片
    parser.add_argument("--view", action="store_true",
                        help="Whether to visualize results during validation")
    
    # 验证模式：如果设置了这个参数，就只做测试不做训练
    parser.add_argument("--validate", action="store_true",
                        help="Whether to evaluate the checkpoint")
    
    # GPU设置：你想用哪个GPU来训练模型（默认用第0个）
    parser.add_argument("--gpus", nargs='+', type=int, default=[0, ],
                        help="Used GPU indices")
    
    # 随机种子：为了让实验结果可以重现，设置一个固定的数字
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    args = parser.parse_args()  # 解析终端输入的所有参数

    return args  # 返回解析好的参数对象


def main():
    """主函数：程序的入口点，这里包含了整个程序的主要逻辑"""
    # 第一步：获取用户输入的所有参数
    args = parse_args()
    
    # 第二步：设置程序可以使用的GPU设备
    # 例如你有两个GPU，只想用第0个和第1个，就设置为 "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu) for gpu in args.gpus)
    print("CUDA_VISIBLE_DEVICES: ",os.environ.get("CUDA_VISIBLE_DEVICES"))


    # 第三步：加载配置文件（这个文件包含了所有训练设置）
    cfg = Config.fromfile(args.config)
    
    # 第四步：用命令行参数覆盖配置文件中的设置
    cfg.gpus = len(args.gpus)  # 设置使用的GPU数量
    cfg.load_from = args.load_from  # 设置要加载的模型文件路径
    cfg.view = args.view  # 设置是否在测试时显示图片
    cfg.seed = args.seed  # 设置随机种子（保证结果可重现）
    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs  # 设置工作目录

    # 第五步：启用cuDNN自动优化（这能让GPU计算更快）
    cudnn.benchmark = True

    # 第六步：创建训练/测试的执行器（这是整个程序的核心）
    runner = Runner(cfg)
    
    # 第七步：根据用户设置决定是做训练还是测试
    if args.validate:  # 如果用户指定了 --validate 参数
        runner.validate()  # 就只做测试，不训练模型
    else:  # 否则
        runner.train()  # 就做训练（训练中会定期做测试）


if __name__ == "__main__":  # 当你在终端直接运行这个文件时，这行代码会被执行
    main()  # 调用主函数，开始整个程序的运行
