from typing import Dict, Any, Callable, List
from torch.utils.data import Dataset


DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {}


def register_dataset(name: str):
    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str, **kwargs: Any) -> Dataset:
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available}")
    return DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> List[str]:
    return list(DATASET_REGISTRY.keys())
