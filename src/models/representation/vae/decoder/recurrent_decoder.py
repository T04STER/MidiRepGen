import torch
from src.models.representation.vae.decoder.vae_decoder import VaeDecoder
from torch import nn

class LSTMVaeDecoder(VaeDecoder):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, output_dim: int, input_dim: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.start_token = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, z: torch.Tensor, seq_length: int) -> torch.Tensor:
        batch_size = z.size(0)
        h0 = self.latent_to_hidden(z).unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]
        c0 = self.latent_to_cell(z).unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]

        input_token = self.start_token.unsqueeze(0).repeat(batch_size, 1, 1)

        outputs = []
        hidden = (h0, c0)

        for _ in range(seq_length):
            out, hidden = self.lstm(input_token, hidden)
            out = out[:, -1, :]
            decoded = self.fc_out(out)
            outputs.append(decoded.unsqueeze(1))
            input_token = decoded.unsqueeze(1)
        return torch.cat(outputs, dim=1)
    

class LSTMVaeDecoderWithTeacherForcing(LSTMVaeDecoder):

    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, output_dim: int, input_dim: int, teacher_forcing_ratio: float = 0.5):
        super().__init__(latent_dim, hidden_dim, num_layers, output_dim, input_dim)
        self.teacher_forcing_ratio = teacher_forcing_ratio


    def forward(self, z: torch.Tensor, seq_length: int, true_output=None) -> torch.Tensor:
        batch_size = z.size(0)
        h0 = self.latent_to_hidden(z).unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]
        c0 = self.latent_to_cell(z).unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch_size, hidden_dim]

        input_token = self.start_token.unsqueeze(0).repeat(batch_size, 1, 1)

        outputs = []
        hidden = (h0, c0)

        for _ in range(seq_length):
            out, hidden = self.lstm(input_token, hidden)
            out = out[:, -1, :]
            decoded = self.fc_out(out)
            outputs.append(decoded.unsqueeze(1))
            input_token = decoded.unsqueeze(1)
            if true_output is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                input_token = true_output[:, len(outputs) - 1, :].unsqueeze(1)
            else:
                input_token = decoded.unsqueeze(1)
        return torch.cat(outputs, dim=1)
