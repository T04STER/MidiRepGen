from typing import Any
import yaml


_DVC_CONFIG_FILE = "params.yaml"


def get_config(key: str | None = None) -> dict[str, Any]:
    with open(_DVC_CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
    if key is not None:
        return config[key]
    return config


def seed_all(seed: int) -> None:
    """
    Set the seed for all random number generators to ensure reproducibility.
    
    :param seed: The seed value to set.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)