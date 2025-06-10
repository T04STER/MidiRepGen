import torch
from src.models.representation.vae.decoder.vae_decoder import VaeDecoder
from torch import nn


class LSTMVaeDecoder(VaeDecoder):
    def __init__(self, latent_dim:int, hidden_dim: int, num_layers: int, output, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, output)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(z)
        out = self.fc_out(lstm_out)
        return out
        
