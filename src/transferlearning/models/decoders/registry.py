from typing import Dict, Any, Callable, List
import torch.nn as nn


DECODER_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_decoder(name: str):
    def decorator(cls):
        if name in DECODER_REGISTRY:
            raise ValueError(f"Decoder '{name}' is already registered")
        DECODER_REGISTRY[name] = cls
        return cls
    return decorator


def get_decoder(name: str, **kwargs: Any) -> nn.Module:
    if name not in DECODER_REGISTRY:
        available = list(DECODER_REGISTRY.keys())
        raise ValueError(f"Decoder '{name}' not found. Available: {available}")
    return DECODER_REGISTRY[name](**kwargs)


def list_decoders() -> List[str]:
    return list(DECODER_REGISTRY.keys())
