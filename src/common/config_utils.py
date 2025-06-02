from typing import Any
import yaml


_DVC_CONFIG_FILE = 'params.yaml'


def get_config(key: str | None = None) -> dict[str, Any]:
    with open (_DVC_CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
    if key is not None:
        return config[key]
    return config