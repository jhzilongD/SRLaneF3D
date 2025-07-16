# 这是ResNet骨干网络的实现文件，ResNet是深度学习中最重要的卷积神经网络之一
# 它解决了深度网络训练中的梯度消失问题，通过残差连接让网络可以训练得更深

# flake8: noqa
from torch import nn  # 导入PyTorch神经网络模块
from torch.hub import load_state_dict_from_url  # 用于从网络下载预训练模型参数

from srlane.models.registry import BACKBONES  # 导入骨干网络注册器

# 预训练模型的下载链接字典
# 这些模型都是在ImageNet数据集上预训练的，可以用于迁移学习
model_urls = {
    "resnet18":  # 18层ResNet，比较轻量，适合快速实验
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34":  # 34层ResNet，在准确率和速度间取得平衡
        "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50":  # 50层ResNet，最常用的版本，性能优异
        "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101":  # 101层ResNet，更深的网络，准确率更高但速度较慢
        "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152":  # 152层ResNet，非常深的网络
        "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d":  # ResNeXt变种，使用分组卷积提高效率
        "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d":  # 更深的ResNeXt版本
        "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2":  # 宽ResNet，增加通道数而不是深度
        "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2":  # 更深的宽ResNet
        "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3卷积层，带填充
    这是ResNet中最常用的卷积层，3x3的卷积核可以捕获局部特征
    """
    return nn.Conv2d(in_planes,     # 输入通道数
                     out_planes,    # 输出通道数
                     kernel_size=3, # 卷积核大小3x3
                     stride=stride, # 步长，控制输出大小
                     padding=dilation, # 填充，保持特征图大小
                     groups=groups, # 分组卷积，用于ResNeXt
                     bias=False,    # 不使用偏置，因为后面有BatchNorm
                     dilation=dilation)  # 膨胀卷积，用于增大感受野


def conv1x1(in_planes, out_planes, stride=1):
    """1x1卷积层
    主要用于改变通道数，不改变特征图的空间大小
    在ResNet中用于瓶颈结构和残差连接的维度匹配
    """
    return nn.Conv2d(in_planes,     # 输入通道数
                     out_planes,    # 输出通道数
                     kernel_size=1, # 1x1卷积核
                     stride=stride, # 步长
                     bias=False)    # 不使用偏置


class BasicBlock(nn.Module):
    """ResNet的基本块，用于ResNet-18和ResNet-34
    这是两层的残差块，结构简单但有效
    """
    expansion = 1  # 输出通道数相对于输入通道数的扩展系数

    def __init__(self,
                 inplanes,      # 输入通道数
                 planes,        # 基本通道数
                 stride=1,      # 步长，用于下采样
                 downsample=None, # 下采样层，用于残差连接的维度匹配
                 groups=1,      # 分组数，基本块只支持普通卷积
                 base_width=64, # 基本宽度
                 dilation=1,    # 膨胀系数
                 norm_layer=None): # 正则化层
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认使用批次正则化
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        # if dilation > 1:
        #     raise NotImplementedError(
        #         "Dilation > 1 not supported in BasicBlock")
        # 第一个卷积层和下采样层都会在stride != 1时对输入进行下采样
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)  # 第一个3x3卷积
        self.bn1 = norm_layer(planes)  # 第一个正则化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，就地操作节省内存
        self.conv2 = conv3x3(planes, planes, dilation=dilation)  # 第二个3x3卷积
        self.bn2 = norm_layer(planes)  # 第二个正则化层
        self.downsample = downsample  # 下采样层，用于残差连接
        self.stride = stride  # 步长

    def forward(self, x):
        """BasicBlock的前向传播函数"""
        identity = x  # 保存输入，用于残差连接

        # 第一个卷积块：卷积 -> 正则化 -> 激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积块：卷积 -> 正则化 （注意这里没有激活）
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，对残差连接的输入进行维度匹配
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：将输入与输出相加，这是ResNet的核心创新
        out += identity
        out = self.relu(out)  # 最后再进行激活

        return out


class Bottleneck(nn.Module):
    """ResNet的瓶颈块，用于ResNet-50及更深的网络
    这是三层的残差块，使用1x1-3x3-1x1的瓶颈结构来减少计算量
    """
    expansion = 4  # 输出通道数是基本通道数的4倍

    def __init__(self,
                 inplanes,      # 输入通道数
                 planes,        # 基本通道数（中间层的通道数）
                 stride=1,      # 步长
                 downsample=None, # 下采样层
                 groups=1,      # 分组数，用于ResNeXt
                 base_width=64, # 基本宽度
                 dilation=1,    # 膨胀系数
                 norm_layer=None): # 正则化层
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认使用批次正则化
        # 计算中间层的通道数，支持ResNeXt和Wide ResNet
        width = int(planes * (base_width / 64.)) * groups
        # 第二个卷积层和下采样层都会在stride != 1时对输入进行下采样
        self.conv1 = conv1x1(inplanes, width)  # 1x1卷积，用于降维
        self.bn1 = norm_layer(width)  # 第一个正则化层
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # 3x3卷积，主要的特征提取
        self.bn2 = norm_layer(width)  # 第二个正则化层
        self.conv3 = conv1x1(width, planes * self.expansion)  # 1x1卷积，用于升维
        self.bn3 = norm_layer(planes * self.expansion)  # 第三个正则化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.downsample = downsample  # 下采样层
        self.stride = stride  # 步长

    def forward(self, x):
        """Bottleneck的前向传播函数"""
        identity = x  # 保存输入，用于残差连接

        # 第一个卷积块：1x1卷积降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积块：3x3卷积提取特征
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三个卷积块：1x1卷积升维 （注意这里没有激活）
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要下采样，对残差连接的输入进行维度匹配
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：将输入与输出相加
        out += identity
        out = self.relu(out)  # 最后再进行激活

        return out


@BACKBONES.register_module  # 将这个类注册为骨干网络
class ResNetWrapper(nn.Module):
    """ResNet的包装器类，适配了SRLane系统的需求
    这个类封装了标准的ResNet，并为车道线检测任务做了一些定制化修改
    """
    def __init__(self,
                 resnet="resnet18",    # ResNet的类型，默认使用ResNet-18
                 pretrained=True,     # 是否使用预训练模型
                 replace_stride_with_dilation=[False, False, False], # 是否用膨胀卷积替换步长
                 out_conv=False,      # 是否添加输出卷积层
                 fea_stride=8,        # 特征步长
                 out_channel=128,     # 输出通道数
                 in_channels=[64, 128, 256, 512], # 各层的通道数配置
                 cfg=None):           # 配置对象
        super(ResNetWrapper, self).__init__()
        self.cfg = cfg  # 保存配置对象
        self.in_channels = in_channels  # 保存通道数配置

        # 使用eval函数动态创建指定类型的ResNet
        self.model = eval(resnet)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            in_channels=self.in_channels)
        self.out = None  # 输出卷积层初始化为空
        if out_conv:  # 如果需要添加输出卷积层
            out_channel = 512  # 默认输出通道数
            # 从后往前找到第一个有效的通道数
            for chan in reversed(self.in_channels):
                if chan < 0: continue  # 跳过无效的通道数
                out_channel = chan
                break
            # 创建1x1卷积层调整输出通道数
            self.out = conv1x1(out_channel * self.model.expansion,
                               cfg.featuremap_out_channel)

    def forward(self, x):
        """ResNetWrapper的前向传播函数"""
        x = self.model(x)  # 通过ResNet模型得到多层特征
        if self.out:  # 如果有输出卷积层
            x[-1] = self.out(x[-1])  # 对最后一层特征进行输出卷积
        return x  # 返回多层特征列表

    def __repr__(self):
        """ResNetWrapper的字符串表示，显示参数数量"""
        num_params = sum(map(lambda x: x.numel(), self.parameters()))  # 计算总参数数量
        return f"#Params of {self._get_name()}: {num_params / 10 ** 6:<.2f}[M]"  # 返回参数信息


class ResNet(nn.Module):
    """ResNet的主类，实现了整个网络结构
    这个类实现了标准的ResNet网络，并为车道线检测任务做了一些调整
    """
    def __init__(self,
                 block,           # 使用的块类型（BasicBlock或Bottleneck）
                 layers,          # 各层的块数量列表
                 zero_init_residual=False,  # 是否对残差分支进行零初始化
                 groups=1,        # 分组数，用于ResNeXt
                 width_per_group=64,  # 每组的宽度
                 replace_stride_with_dilation=None,  # 是否用膨胀卷积替换步长
                 norm_layer=None, # 正则化层类型
                 in_channels=None):  # 各层的输入通道数
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认使用批次正则化
        self._norm_layer = norm_layer

        self.inplanes = 64  # 初始通道数
        self.dilation = 1   # 初始膨胀系数
        if replace_stride_with_dilation is None:
            # 每个元素表示是否用膨胀卷积替换步长为2的卷积
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups  # 分组数
        self.base_width = width_per_group  # 基本宽度
        
        # 第一个卷积层：7x7卷积，步长为2，输入为3通道的RGB图像
        self.conv1 = nn.Conv2d(3,              # 输入通道数（RGB图像）
                               self.inplanes,  # 输出通道数
                               kernel_size=7,  # 卷积核大小
                               stride=2,       # 步长，将图像大小减半
                               padding=3,      # 填充
                               bias=False)     # 不使用偏置
        self.bn1 = norm_layer(self.inplanes)  # 正则化层
        self.relu = nn.ReLU(inplace=True)     # 激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层
        
        self.in_channels = in_channels  # 保存通道数配置
        # 构建四个残差块组
        self.layer1 = self._make_layer(block, in_channels[0], layers[0])
        self.layer2 = self._make_layer(block,
                                       in_channels[1],
                                       layers[1],
                                       stride=2,  # 步长为2，进行下采样
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       in_channels[2],
                                       layers[2],
                                       stride=2,  # 步长为2，进行下采样
                                       dilate=replace_stride_with_dilation[1])
        if in_channels[3] > 0:  # 只有当通道数大于0时才构建第4层
            self.layer4 = self._make_layer(
                block,
                in_channels[3],
                layers[3],
                stride=2,  # 步长为2，进行下采样
                dilate=replace_stride_with_dilation[2])
        self.expansion = block.expansion  # 保存扩展系数

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        """ResNet的前向传播函数，返回多层特征"""
        # 第一阶段：初始特征提取
        x = self.conv1(x)   # 7x7卷积，将图像大小减半
        x = self.bn1(x)     # 正则化
        x = self.relu(x)    # 激活
        x = self.maxpool(x) # 最大池化，再次减半

        # 第二阶段：通过四个残差块组提取特征
        out_layers = []  # 存储每一层的输出特征
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            if not hasattr(self, name):  # 检查层是否存在
                continue
            layer = getattr(self, name)  # 获取对应的层
            x = layer(x)  # 通过层进行特征提取
            out_layers.append(x)  # 保存输出特征

        return out_layers  # 返回多层特征列表，用于特征金字塔网络


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    """ResNet的通用构建函数
    这个函数可以构建不同类型的ResNet模型，并可选地加载预训练模型
    """
    model = ResNet(block, layers, **kwargs)  # 创建模型
    if pretrained:  # 如果需要预训练模型
       # print("pretrained model: ", model_urls[arch])
        state_dict = load_state_dict_from_url(model_urls[arch])  # 下载预训练模型参数
        model_state_dict = model.state_dict()  # 获取当前模型参数
        # 只加载形状匹配的参数，忽略不匹配的参数
        state_dict = {k: v for k, v in state_dict.items() if
                      k in model_state_dict and v.shape == model_state_dict[k].shape}
        model.load_state_dict(state_dict, strict=False)  # 加载参数
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)
