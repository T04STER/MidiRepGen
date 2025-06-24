from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch
from torch import nn


class VaeEncoder(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for VAE encoders.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded representation.
        """
        ...