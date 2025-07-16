# SRLane Version Information

## File Overview

This module defines the version information for the SRLane (Sketch and Refine Lane Detection) project. It provides a centralized location for version management and ensures consistent version reporting across the entire codebase. The version information is used for package management, compatibility checking, and release tracking.

## Version Definition

```python
__version__ = "1.0"
```

**Variable**: `__version__`

**Purpose**: Defines the current version of the SRLane package following semantic versioning conventions.

**Current Version**: "1.0"
- Indicates the first stable release of the SRLane implementation
- Represents a complete, production-ready version of the lane detection system

## Version Format and Conventions

### Semantic Versioning

The SRLane project follows semantic versioning (SemVer) principles:

**Format**: `MAJOR.MINOR.PATCH`

**Version Components**:
- **MAJOR** (1): Incremented for incompatible API changes
- **MINOR** (0): Incremented for backwards-compatible functionality additions  
- **PATCH** (implicit 0): Incremented for backwards-compatible bug fixes

**Current State**: Version "1.0" can be interpreted as "1.0.0"

### Version Significance

**Version 1.0 Milestone**:
- **API Stability**: Core interfaces are considered stable
- **Feature Complete**: All planned features for initial release are implemented
- **Production Ready**: Suitable for research and practical applications
- **Documentation**: Comprehensive documentation and examples provided

## Integration with Python Packaging

### Package Metadata

The version defined here integrates with Python's packaging system:

```python
# In setup.py (typical usage)
from srlane.version import __version__

setup(
    name="srlane",
    version=__version__,
    description="Sketch and Refine Lane Detection",
    # ... other metadata
)
```

### Runtime Version Access

```python
# Users can check version programmatically
import srlane
print(f"SRLane version: {srlane.__version__}")

# Or directly from version module
from srlane.version import __version__
print(f"Version: {__version__}")
```

## Usage in SRLane System

### Configuration and Logging

```python
# In training/evaluation scripts
from srlane.version import __version__
import logging

logger = logging.getLogger(__name__)
logger.info(f"Starting SRLane v{__version__}")

# In configuration files
version_info = {
    "srlane_version": __version__,
    "commit_hash": "...",  # Could be added
    "build_date": "...",   # Could be added
}
```

### Experiment Tracking

```python
# Integration with experiment recording
class Recorder:
    def __init__(self, cfg):
        self.version = __version__
        self.logger.info(f"SRLane version: {self.version}")
        
        # Save version info with experiment
        self.write(f"Experiment started with SRLane v{self.version}")
```

### Model Checkpoints

```python
# Include version in saved models
def save_model(net, recorder):
    checkpoint = {
        "net": net.state_dict(),
        "srlane_version": __version__,
        "epoch": recorder.epoch,
        # ... other metadata
    }
    torch.save(checkpoint, checkpoint_path)
```

## Future Version Management

### Expansion Possibilities

The version module could be expanded to include:

```python
# Extended version information
__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
__build__ = "20240716"
__commit__ = "b00b741"  # Git commit hash
__author__ = "SRLane Team"
__email__ = "contact@srlane.org"
```

### Development Versions

```python
# Development version format
__version__ = "1.1.0-dev"  # Development version
__version__ = "1.0.1-rc1"  # Release candidate
__version__ = "1.0.0-alpha" # Alpha version
```

### Version Comparison

```python
# Version comparison utilities
def parse_version(version_str):
    """Parse version string into comparable tuple."""
    return tuple(map(int, version_str.split('.')))

def is_compatible(required_version):
    """Check if current version meets requirement."""
    current = parse_version(__version__)
    required = parse_version(required_version)
    return current >= required
```

## Integration Examples

### Library Compatibility

```python
# Check minimum SRLane version
try:
    from srlane.version import __version__
    if parse_version(__version__) < parse_version("1.0.0"):
        raise ImportError("SRLane >= 1.0.0 required")
except ImportError:
    raise ImportError("SRLane not installed or version too old")
```

### Feature Detection

```python
# Version-based feature availability
def has_feature(feature_name):
    """Check if feature is available in current version."""
    feature_versions = {
        "cascade_refinement": "1.0.0",
        "multi_gpu_training": "1.0.0", 
        "tensorboard_logging": "1.0.0",
        "advanced_augmentation": "1.1.0",  # Future feature
    }
    
    if feature_name not in feature_versions:
        return False
        
    required = parse_version(feature_versions[feature_name])
    current = parse_version(__version__)
    return current >= required
```

### Documentation Generation

```python
# Automatic version in documentation
def generate_docs():
    """Generate documentation with current version."""
    doc_template = f"""
    # SRLane Documentation v{__version__}
    
    This documentation covers SRLane version {__version__}.
    For other versions, see the version-specific documentation.
    """
    return doc_template
```

## Best Practices

### Version Update Process

1. **Single Source**: Update only this file for version changes
2. **Consistent Format**: Maintain semantic versioning format
3. **Documentation**: Update changelog and documentation with version changes
4. **Testing**: Verify version compatibility across all modules

### Deployment Considerations

- **Reproducibility**: Include version in all experiment records
- **Compatibility**: Check version compatibility in dependent projects
- **Distribution**: Ensure version is properly packaged with releases
- **Debugging**: Include version information in error reports

### Development Workflow

```python
# Pre-release checklist
def validate_release():
    """Validate version before release."""
    checks = [
        "Version format is valid",
        "Documentation updated", 
        "Tests pass",
        "Changelog updated",
        "API compatibility verified"
    ]
    # ... validation logic
```

## Technical Considerations

### Import Performance

The minimal version module ensures:
- **Fast Import**: No expensive operations during import
- **No Dependencies**: Doesn't require other SRLane modules
- **Memory Efficient**: Minimal memory footprint

### Thread Safety

The version string is immutable and thread-safe:
- Multiple threads can safely read the version
- No synchronization required for version access
- Safe for use in multi-threaded training environments

This simple but essential module provides the foundation for version management and compatibility tracking throughout the SRLane ecosystem.