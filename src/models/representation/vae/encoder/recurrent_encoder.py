from typing import Tuple
from torch import nn
import torch

from src.models.representation.vae.encoder.vae_encoder import VaeEncoder


class LSTMVaeEncoder(VaeEncoder):
    """
    A class representing a VAE encoder that uses LSTM with attention mechanism.
    This class is designed to handle sequential data and extract meaningful features
    using LSTM layers combined with an attention mechanism.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers:int=1):
        super().__init__()

        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mu_linear = nn.Linear(hidden_dim, latent_dim)
        self.logvar_linear = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input data.
        
        Returns:
            Encoded representation of the input data. [batch size, sequence length, latent dimension], [batch size, sequence length, latent dimension]
        """
        _, (h_n, _) = self.lstm(x)
        lstm_out = h_n[-1]
        mu_linear_out = self.mu_linear(lstm_out)
        logvar_linear_out = self.logvar_linear(lstm_out)
        return mu_linear_out, logvar_linear_out
