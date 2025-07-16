# SRLane Registry System

## File Overview

This module implements a flexible registry system for dynamic component registration and instantiation in the SRLane framework. The registry provides a centralized mechanism for managing different types of components (models, datasets, losses, etc.) and supports configuration-driven object creation. This system enables modular architecture and easy experimentation with different component combinations.

## Imports and Dependencies

```python
import six
import inspect
```

**Dependencies**:
- **six**: Python 2/3 compatibility library for string type checking
- **inspect**: Runtime introspection utilities for class validation

## Utility Functions

### String Type Checking

```python
def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)
```

**Function**: `is_str`

**Purpose**: Provides cross-compatible string type checking for Python 2/3.

**Parameters**:
- `x` (any): Object to test for string type

**Returns**: 
- `bool`: True if input is a string type, False otherwise

**Implementation**: Uses `six.string_types` for compatibility across Python versions
- Python 2: Checks for `str` and `unicode`  
- Python 3: Checks for `str`

## Core Registry Class

### Registry Implementation

```python
class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
```

**Class**: `Registry`

**Purpose**: Manages registration and retrieval of components by name, enabling dynamic instantiation from configuration.

**Initialization Parameters**:
- `name` (str): Registry name for identification and debugging

**Internal Structure**:
- `_name`: Registry identifier (e.g., "MODELS", "DATASETS", "LOSSES")
- `_module_dict`: Dictionary mapping component names to classes

### String Representation

```python
def __repr__(self):
    format_str = self.__class__.__name__
    format_str += f"(name={self._name}, "
    format_str += f"items={list(self._module_dict.keys())})"
    return format_str
```

**Method**: `__repr__`

**Purpose**: Provides informative string representation for debugging and inspection.

**Output Format**:
```
Registry(name=MODELS, items=['ResNet', 'TwoStageDetector', 'LocalAngleHead'])
Registry(name=DATASETS, items=['CULaneDataset', 'TuSimpleDataset'])
```

**Benefits**:
- **Debugging**: Easy identification of registered components
- **Development**: Quick overview of available options
- **Documentation**: Self-documenting component availability

### Property Access

```python
@property
def name(self):
    return self._name

@property
def module_dict(self):
    return self._module_dict
```

**Properties**:
- `name`: Read-only access to registry name
- `module_dict`: Read-only access to registered components dictionary

**Purpose**: Provides controlled access to internal state without exposing implementation details.

### Component Retrieval

```python
def get(self, key):
    return self._module_dict.get(key, None)
```

**Method**: `get`

**Purpose**: Safely retrieves a registered component by name.

**Parameters**:
- `key` (str): Component name to retrieve

**Returns**:
- Component class if found, `None` if not registered

**Safe Access**: Uses dict.get() to avoid KeyError exceptions

### Component Registration

```python
def _register_module(self, module_class):
    """Register a module.

    Args:
        module (:obj:`nn.Module`): Module to be registered.
    """
    if not inspect.isclass(module_class):
        raise TypeError(f"module must be a class, "
                        f"but got {type(module_class)}")
    module_name = module_class.__name__
    if module_name in self._module_dict:
        raise KeyError(f"{module_name} already registered in {self.name}")
    self._module_dict[module_name] = module_class
```

**Method**: `_register_module` (Internal)

**Purpose**: Core registration logic with validation and error checking.

**Parameters**:
- `module_class` (class): Class to register in the registry

**Validation Steps**:
1. **Class Validation**: Ensures input is actually a class using `inspect.isclass()`
2. **Name Extraction**: Uses class `__name__` attribute as registry key
3. **Duplicate Check**: Prevents overwriting existing registrations
4. **Registration**: Stores class in internal dictionary

**Error Handling**:
- `TypeError`: Raised if non-class object is provided
- `KeyError`: Raised if component name already exists

### Public Registration Interface

```python
def register_module(self, cls):
    self._register_module(cls)
    return cls
```

**Method**: `register_module`

**Purpose**: Public interface for component registration, designed for use as decorator.

**Parameters**:
- `cls` (class): Class to register

**Returns**: 
- `cls`: Same class (enables decorator usage)

**Decorator Pattern**: Returns the class unchanged, allowing use as decorator:
```python
@registry.register_module
class MyComponent:
    pass
```

## Configuration-Based Object Creation

### Build Function

```python
def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
```

**Function**: `build_from_cfg`

**Purpose**: Creates objects dynamically from configuration dictionaries using registry lookup.

**Parameters**:
- `cfg` (dict): Configuration dictionary specifying object type and parameters
- `registry` (Registry): Registry to search for component types
- `default_args` (dict, optional): Default arguments merged with config

**Returns**: Instantiated object of the specified type

### Configuration Validation

```python
assert isinstance(cfg, dict) and "type" in cfg
assert isinstance(default_args, dict) or default_args is None
```

**Validation Requirements**:
1. **Config Type**: Must be dictionary
2. **Type Key**: Must contain "type" field specifying component name
3. **Default Args**: Must be dictionary or None

**Example Valid Config**:
```python
cfg = {
    "type": "ResNet",
    "depth": 50,
    "num_classes": 1000,
    "pretrained": True
}
```

### Type Resolution

```python
args = cfg.copy()
obj_type = args.pop("type")
if is_str(obj_type):
    obj_cls = registry.get(obj_type)
    if obj_cls is None:
        raise KeyError(f"{obj_type} not in the {registry.name} registry")
elif inspect.isclass(obj_type):
    obj_cls = obj_type
else:
    raise TypeError(f"type must be a str or valid type, "
                    f"but got {type(obj_type)}")
```

**Type Resolution Process**:
1. **Config Copy**: Creates working copy to avoid modifying original
2. **Type Extraction**: Removes "type" field from arguments
3. **String Lookup**: If type is string, looks up in registry
4. **Direct Class**: If type is already a class, uses directly
5. **Error Handling**: Raises appropriate errors for invalid types

**Supported Type Formats**:
- **String**: `"ResNet"` → Look up in registry
- **Class**: `ResNet` → Use directly
- **Invalid**: Anything else raises TypeError

### Argument Merging

```python
if default_args is not None:
    for name, value in default_args.items():
        args.setdefault(name, value)
```

**Argument Merging Logic**:
- Uses `setdefault()` to add defaults without overwriting config values
- Config arguments take precedence over defaults
- Enables reusable component configurations

**Example**:
```python
cfg = {"type": "ResNet", "depth": 50}
default_args = {"num_classes": 1000, "depth": 101}
# Result: {"depth": 50, "num_classes": 1000}  # Config depth wins
```

### Object Instantiation

```python
return obj_cls(**args)
```

**Instantiation**: Uses Python's `**kwargs` expansion to pass arguments to constructor.

**Process Flow**:
```
Config Dict → Type Resolution → Argument Merging → Class(**args) → Object Instance
```

## Integration with SRLane System

### Registry Setup

```python
# Typical SRLane registry definitions
MODELS = Registry('MODELS')
DATASETS = Registry('DATASETS') 
LOSSES = Registry('LOSSES')
HEADS = Registry('HEADS')
BACKBONES = Registry('BACKBONES')
```

### Component Registration

```python
# Model registration
@MODELS.register_module
class TwoStageDetector(nn.Module):
    def __init__(self, backbone, neck, head):
        # ... implementation

@BACKBONES.register_module  
class ResNet(nn.Module):
    def __init__(self, depth, num_classes):
        # ... implementation

@DATASETS.register_module
class CULaneDataset(Dataset):
    def __init__(self, data_root, ann_file):
        # ... implementation
```

### Configuration-Driven Building

```python
# In configuration files
model_cfg = dict(
    type='TwoStageDetector',
    backbone=dict(type='ResNet', depth=50),
    neck=dict(type='ChannelMapper', in_channels=[256, 512, 1024]),
    head=dict(type='LocalAngleHead', num_classes=2)
)

# Building from config
model = build_from_cfg(model_cfg, MODELS)
```

## Usage Examples

### Basic Registry Usage

```python
from srlane.registry import Registry, build_from_cfg

# Create registry
MODELS = Registry('MODELS')

# Register components
@MODELS.register_module
class SimpleModel:
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

# Build from config
config = {
    "type": "SimpleModel",
    "hidden_size": 256,
    "num_layers": 4
}

model = build_from_cfg(config, MODELS)
print(f"Model: {model.hidden_size}, {model.num_layers}")
```

### Complex Hierarchical Building

```python
# Nested configuration
config = {
    "type": "TwoStageDetector",
    "backbone": {
        "type": "ResNet", 
        "depth": 50,
        "pretrained": True
    },
    "head": {
        "type": "LocalAngleHead",
        "in_channels": 256,
        "num_classes": 2
    }
}

# Build with nested components
def build_detector(cfg):
    backbone = build_from_cfg(cfg['backbone'], BACKBONES)
    head = build_from_cfg(cfg['head'], HEADS)
    
    detector_cfg = cfg.copy()
    detector_cfg['backbone'] = backbone
    detector_cfg['head'] = head
    
    return build_from_cfg(detector_cfg, MODELS)

detector = build_detector(config)
```

### Default Arguments

```python
# Default configuration for all models
default_model_args = {
    "init_weights": True,
    "device": "cuda",
    "precision": "fp32"
}

# Build with defaults
model = build_from_cfg(
    {"type": "ResNet", "depth": 50},
    MODELS,
    default_args=default_model_args
)
```

### Registry Inspection

```python
# View available components
print(MODELS)  # Registry(name=MODELS, items=['ResNet', 'TwoStageDetector'])

# Check if component exists
if MODELS.get('ResNet') is not None:
    print("ResNet is available")

# Get component class directly
ResNetClass = MODELS.get('ResNet')
model = ResNetClass(depth=50)
```

## Advanced Features

### Conditional Registration

```python
def register_if_available(registry, cls):
    """Register class only if dependencies are available."""
    try:
        # Test if class can be imported/used
        cls._test_dependencies()
        registry.register_module(cls)
    except ImportError:
        print(f"Skipping {cls.__name__} due to missing dependencies")
```

### Registry Inheritance

```python
class SpecializedRegistry(Registry):
    def register_module(self, cls):
        # Add custom validation
        if not hasattr(cls, 'required_method'):
            raise ValueError(f"{cls.__name__} missing required_method")
        return super().register_module(cls)
```

### Multi-Registry Building

```python
def build_complex_object(cfg):
    """Build object that requires multiple registries."""
    backbone = build_from_cfg(cfg['backbone'], BACKBONES)
    neck = build_from_cfg(cfg['neck'], NECKS) 
    head = build_from_cfg(cfg['head'], HEADS)
    
    model_cfg = cfg.copy()
    model_cfg.update({
        'backbone': backbone,
        'neck': neck, 
        'head': head
    })
    
    return build_from_cfg(model_cfg, MODELS)
```

## Benefits and Design Principles

### Modularity
- **Loose Coupling**: Components don't need to know about each other
- **Pluggable Architecture**: Easy to swap implementations
- **Extensibility**: New components added without modifying existing code

### Configuration-Driven Development
- **Reproducibility**: Experiments fully specified by config files
- **Flexibility**: Same code supports many different configurations  
- **Maintainability**: Changes through configuration instead of code

### Type Safety
- **Runtime Validation**: Ensures proper types and required fields
- **Clear Error Messages**: Informative errors for debugging
- **Documentation**: Registry contents self-document available options

This registry system provides the foundation for SRLane's modular, configurable architecture, enabling flexible experimentation and clean separation of concerns.