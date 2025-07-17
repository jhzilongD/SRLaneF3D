import six  # 导入six库，用于Python2和Python3的兼容性处理
import inspect  # 导入inspect模块，用于检查对象类型和属性


def is_str(x):  # 定义判断字符串的辅助函数
    """Whether the input is an string instance."""  # 判断输入是否为字符串实例
    return isinstance(x, six.string_types)  # 使用six库兼容不同Python版本的字符串类型检查


class Registry(object):  # 定义注册器类，用于管理和注册模块
    def __init__(self, name):  # 初始化方法
        self._name = name  # 存储注册器的名称
        self._module_dict = dict()  # 创建空字典用于存储注册的模块

    def __repr__(self):  # 定义对象的字符串表示方法
        format_str = self.__class__.__name__  # 获取类名
        format_str += f"(name={self._name}, "  # 添加注册器名称
        format_str += f"items={list(self._module_dict.keys())})"  # 添加已注册模块的键列表
        return format_str  # 返回格式化的字符串

    @property  # 将name定义为只读属性
    def name(self):  # 获取注册器名称的方法
        return self._name  # 返回注册器名称

    @property  # 将module_dict定义为只读属性
    def module_dict(self):  # 获取模块字典的方法
        return self._module_dict  # 返回存储模块的字典

    def get(self, key):  # 根据键获取注册的模块
        return self._module_dict.get(key, None)  # 返回对应的模块，如果不存在返回None

    def _register_module(self, module_class):  # 内部方法，用于注册模块
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):  # 检查传入的是否为类
            raise TypeError(f"module must be a class, "  # 如果不是类，抛出类型错误
                            f"but got {type(module_class)}")
        module_name = module_class.__name__  # 获取类的名称
        if module_name in self._module_dict:  # 检查是否已经注册过
            raise KeyError(f"{module_name} already registered in {self.name}")  # 如果已注册，抛出键错误
        self._module_dict[module_name] = module_class  # 将模块类存储到字典中

    def register_module(self, cls):  # 公开的注册方法，可作为装饰器使用
        self._register_module(cls)  # 调用内部注册方法
        return cls  # 返回原类，使其可以作为装饰器


def build_from_cfg(cfg, registry, default_args=None):  # 根据配置构建模块的函数
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and "type" in cfg  # 确保cfg是字典且包含"type"键
    assert isinstance(default_args, dict) or default_args is None  # 确保default_args是字典或None
    args = cfg.copy()  # 复制配置字典，避免修改原始配置
    obj_type = args.pop("type")  # 取出并移除"type"键的值
    if is_str(obj_type):  # 如果type是字符串
        obj_cls = registry.get(obj_type)  # 从注册器中获取对应的类
        if obj_cls is None:  # 如果没找到对应的类
            raise KeyError(f"{obj_type} not in the {registry.name} registry")  # 抛出键错误
    elif inspect.isclass(obj_type):  # 如果type本身就是一个类
        obj_cls = obj_type  # 直接使用该类
    else:  # 如果既不是字符串也不是类
        raise TypeError(f"type must be a str or valid type, "  # 抛出类型错误
                        f"but got {type(obj_type)}")
    if default_args is not None:  # 如果提供了默认参数
        for name, value in default_args.items():  # 遍历默认参数
            args.setdefault(name, value)  # 如果args中没有该参数，则使用默认值
    return obj_cls(**args)  # 使用参数实例化类并返回对象
