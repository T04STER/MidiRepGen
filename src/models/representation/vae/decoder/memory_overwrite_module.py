from torch import nn
import torch


class MemoryOverwriteModule(nn.Module):
    """
    A module used to overwrite memory in a decoder autoregresion model.
    Injects latent features into the decoder's memory at each t=1,...T time step.
    Basically forget and input gate
    """
    def __init__(self, latent_input_dim, memory_dim,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forget_gate = nn.Sequential(
            nn.Linear(latent_input_dim, memory_dim),
            nn.Sigmoid()
        )
        self.overwrite_sig = nn.Sequential(
            nn.Linear(latent_input_dim, memory_dim),
            nn.Sigmoid()
        )

        self.overwrite_tanh = nn.Sequential(
            nn.Linear(latent_input_dim, memory_dim),
            nn.Tanh()
        )

    def forward(self, lstm_state, lstm_hidden, latent_state, latent_hidden):
        forgotten_state = torch.mul(lstm_state, self.forget_gate(latent_state))
        forgotten_hidden = torch.mul(lstm_hidden, self.forget_gate(latent_hidden))

        state_input = torch.mul(self.overwrite_sig(latent_state), self.overwrite_tanh(latent_state))
        hidden_input = torch.mul(self.overwrite_sig(latent_hidden), self.overwrite_tanh(latent_hidden))
        new_state = forgotten_state + state_input
        new_hidden = forgotten_hidden + hidden_input
        return new_state, new_hidden


        
        
