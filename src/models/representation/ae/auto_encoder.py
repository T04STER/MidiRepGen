from typing import Optional, Tuple
from torch import nn
import torch

from src.models.representation.vae.decoder.memory_overwrite_module import MemoryOverwriteModule

class Autoencoder(nn.Module):
    """
        Sequential autencoder used to learn a compressed representation of the input data.
        Similar to VAE implementation but doesn't apply teacher forcing
        during training and probabilistic representation. 
    """
    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, t):
        x = x.contiguous()
        z = self.encoder(x, t)        
        x_reconstructed = self.decoder(z, seq_length=x.size(1))
        return x_reconstructed
    

class Encoder(nn.Module):
    """
    Encoder part of the autoencoder.
    It takes input data and encodes it into a compressed representation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers:int=1, dropout: float=0.1, num_timesteps = 1000):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.diff_timestep_embedding = nn.Embedding(num_timesteps, hidden_dim) # injection via hidden state
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim*2, latent_dim)
        

    def forward(self, x: torch.Tensor, diff_timestep: torch.Tensor) -> torch.Tensor:
        hidden_embedding = self.diff_timestep_embedding(diff_timestep).unsqueeze(0).repeat(self.num_layers * 2, 1, 1)
        _, (h_n, _) = self.lstm(x, (hidden_embedding, hidden_embedding))
        lstm_backward = h_n[-1]
        lstm_forward = h_n[-2]
        encoded = self.linear(torch.cat((lstm_forward, lstm_backward), dim=-1))
        return encoded


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, output_dim: int, input_dim: Optional[int]=None):
        super().__init__()
        if input_dim is None:
            input_dim = output_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.num_layers = num_layers
        self.mom = MemoryOverwriteModule(latent_input_dim=hidden_dim, memory_dim=hidden_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.start_token = nn.Parameter(torch.zeros(1, input_dim))

    def lattent_to_h0_c0(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert latent vector z to hidden and cell states for LSTM.
        """
        z = z.unsqueeze(0)
        h0 = torch.tanh(self.latent_to_hidden(z)).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim] 
        c0 = torch.tanh(self.latent_to_cell(z)).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]  
        return h0, c0

    def overwrite_memory(self, lstm_out, latent) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mom(lstm_out[0], lstm_out[1], latent[0], latent[1])

    def autoregressive_step(self, input_token: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], hidden_z: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single step of the LSTM and decode the output.
        """
        out, hidden = self.lstm(input_token, hidden)
        out = out[:, -1, :]
        real_out = self.fc_out(out).unsqueeze(1)
        hidden = self.overwrite_memory(hidden, hidden_z)
        return real_out, hidden

    # type: ignore
    def forward(self, z: torch.Tensor, seq_length: int):
        batch_size = z.size(0)
        out: list[torch.Tensor] = []
        hidden_z = self.lattent_to_h0_c0(z)
        hidden = hidden_z

        input_token = self.start_token.unsqueeze(0).repeat(batch_size, 1, 1)
        for t in range(seq_length):
            input_token, hidden = self.autoregressive_step(input_token, hidden, hidden_z)
            out.append(input_token)
        
        out =  torch.cat(out, dim=1)
        return out
            
        
    