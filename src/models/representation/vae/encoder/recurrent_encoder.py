from typing import Tuple
from torch import nn
import torch

from src.models.representation.vae.encoder.vae_encoder import VaeEncoder


class LSTMAtentionVaeEncoder(VaeEncoder):
    """
    A class representing a VAE encoder that uses LSTM with attention mechanism.
    This class is designed to handle sequential data and extract meaningful features
    using LSTM layers combined with an attention mechanism.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # TODO: THINK ABOUT ATTENTION MECHANISM
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input data.
        
        Returns:
            Encoded representation of the input data.
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        