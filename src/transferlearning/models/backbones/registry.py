from typing import Dict, Any, Callable, List
import torch.nn as nn


BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_backbone(name: str):
    def decorator(cls):
        if name in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone '{name}' is already registered")
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


def get_backbone(name: str, **kwargs: Any) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        available = list(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Backbone '{name}' not found. Available: {available}")
    return BACKBONE_REGISTRY[name](**kwargs)


def list_backbones() -> List[str]:
    return list(BACKBONE_REGISTRY.keys())
