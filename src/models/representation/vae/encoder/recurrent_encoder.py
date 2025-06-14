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


class LSTMVaeEncoderPitchEmbedding(VaeEncoder):
    """
    A class representing a VAE encoder that uses LSTM with attention mechanism.
    This class is designed to handle sequential data and extract meaningful features
    using LSTM layers combined with an attention mechanism.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers:int=1, pitch_embedding_dim: int = 128, linear_scaling_dim: int=4, dropout: float=0.1):
        super().__init__()

        self.linear = nn.Linear(input_dim-1, linear_scaling_dim)
        self.pitch_embedding = nn.Embedding(128, pitch_embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=linear_scaling_dim + pitch_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.mu_linear = nn.Linear(hidden_dim*2, latent_dim)
        self.logvar_linear = nn.Linear(hidden_dim*2, latent_dim)
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input data.
        
        Returns:
            Encoded representation of the input data. [batch size, sequence length, latent dimension], [batch size, sequence length, latent dimension]
        """
        pitch_embeddings = self.pitch_embedding(x[:, :, 0].long())
        # print(f"Pitch embeddings shape: {pitch_embeddings.shape}")
        linear_out = self.linear(x[:, :, 1:]).relu_()
        # print(f"Linear out shape: {linear_out.shape}")
        x = torch.cat((linear_out, pitch_embeddings), dim=-1)
        # print(f"Concatenated input shape: {x.shape}")
        x = x.contiguous()
        _, (h_n, _) = self.lstm(x)
        # print(f"LSTM output shape: {h_n.shape}")
        lstm_out_forward = h_n[-2]
        lstm_out_backward = h_n[-1]
        lstm_out = torch.cat((lstm_out_forward, lstm_out_backward), dim=-1)
        # print(f"LSTM last hidden state shape: {lstm_out.shape}")
        mu_linear_out = self.mu_linear(lstm_out)
        # print(f"Mu linear output shape: {mu_linear_out.shape}")
        logvar_linear_out = self.logvar_linear(lstm_out)
        # print(f"Logvar linear output shape: {logvar_linear_out.shape}")
        return mu_linear_out, logvar_linear_out
