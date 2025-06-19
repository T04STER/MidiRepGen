
from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

LOG_DEFAULT_DIT = "logs"


def get_writter(name: str, prefix: Optional[str]=None) -> SummaryWriter:
    log_dir = Path(LOG_DEFAULT_DIT)
    if prefix:
        log_dir = log_dir / prefix
    log_dir = log_dir / name
    log_dir = log_dir.resolve()
    return SummaryWriter(log_dir=log_dir)