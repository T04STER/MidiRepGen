from typing import Optional
from torch import nn
import torch

from src.models.representation.vae.decoder.memory_overwrite_module import MemoryOverwriteModule
from src.models.representation.vae.decoder.recurrent_decoder import LSTMVaeDecoder


class MomVaeDecoder(LSTMVaeDecoder):

    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, output_dim: int, input_dim: Optional[int]=None, teacher_forcing_ratio: float = 0.5, teacher_forcing_decrease: float = 0.1):
        if input_dim is None:
            input_dim = output_dim
        self.one_hot_pitch_and_other_dim = 128 + 3  # 128 pitches + 3 other dimensions (velocity, duration, etc.)
        super().__init__(latent_dim, hidden_dim, num_layers, output_dim, self.one_hot_pitch_and_other_dim)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decrease = teacher_forcing_decrease
        self.pitch_linear = nn.Linear(output_dim, 128) # outpts as pitch to one-hot
        self.other_scaling = nn.Linear(output_dim, 3)
        self.mom = MemoryOverwriteModule(latent_input_dim=hidden_dim, memory_dim=hidden_dim)

    def step_teacher_forcing(self):
        """
        Decrease the teacher forcing ratio by a fixed amount.
        """
        self.teacher_forcing_ratio = max(0, self.teacher_forcing_ratio - self.teacher_forcing_decrease)

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

    def autoregressive_step(self, input_token: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], hidden_z: tuple[torch.Tensor, torch.Tensor]) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform a single step of the LSTM and decode the output.
        """
        out, hidden = self.lstm(input_token, hidden)
        out = out[:, -1, :]
        out_intermediate = torch.relu(self.fc_out(out))
        pitch = self.pitch_linear(out_intermediate)
        other = nn.functional.softplus(self.other_scaling(out_intermediate))
        hidden = self.overwrite_memory(hidden, hidden_z)
        return (pitch, other), hidden

    # type: ignore
    def forward(self, z: torch.Tensor, seq_length: int, true_output=None):
        batch_size = z.size(0)
        pitches: list[torch.Tensor] = []
        others: list[torch.Tensor] = []
        hidden_z = self.lattent_to_h0_c0(z)
        hidden = hidden_z

        input_token = self.start_token.unsqueeze(0).repeat(batch_size, 1, 1)

        true_output = self.true_output_to_lstm_input(true_output) if true_output is not None and self.teacher_forcing_ratio > 0 else None
        for t in range(seq_length):
            (pitch, other), hidden = self.autoregressive_step(input_token, hidden, hidden_z)
            input_token = torch.cat((pitch, other), dim=-1)  # Concatenate pitch and other dimensions
            input_token = input_token.unsqueeze_(1)
            pitches.append(pitch)
            others.append(other)
            if true_output is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                input_token = true_output[:, t, :].unsqueeze(1)
        
        ptensor = torch.stack(pitches, dim=1)
        otensor = torch.stack(others, dim=1)
        return ptensor, otensor
    
    def true_output_to_lstm_input(self, true_output: torch.Tensor) -> torch.Tensor:
        """
        Convert the true output to the input format for the LSTM.
        """
        ordinal_pitch = true_output[:, :, 0].long()
        pitch_one_hot = torch.nn.functional.one_hot(ordinal_pitch, num_classes=128).float()
        other_features = true_output[:, :, 1:]
        return torch.cat((pitch_one_hot, other_features), dim=-1)