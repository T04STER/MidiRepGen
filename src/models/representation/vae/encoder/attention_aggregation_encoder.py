from typing import Tuple
from torch import nn
import torch


from src.models.representation.vae.encoder.vae_encoder import VaeEncoder


class AttentionAggregatedLSTMVaeEncoder(VaeEncoder):
    """
    DISCLAIMER: WORSE PERFORMANCE THAN LSTMVaeEncoderPitchEmbedding
    A class representing a VAE encoder that uses LSTM with attention mechanism.
    This class is designed to handle sequential data and extract meaningful features
    using LSTM layers combined with an attention mechanism.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers:int=1, pitch_embedding_dim: int = 128, linear_scaling_dim: int=4, dropout: float=0.1, heads_num: int=1):
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
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=1, batch_first=True)
    

    def attention_aggregate(self, all_hiddens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Appplies attention mechanism to aggregate hidden states.
        :param all_hiddens: Tensor of LSTM outputs for each token [batch size, sequence length, hidden dimension]
        :param query: Query tensor for attention. Last output hidden of forward LSTM concated with backward LSTM [batch size, hidden dimension * 2]
        :return: Aggregated hidden states after applying attention [batch size, hidden dimension]
        """
        query = query.unsqueeze(1)  # [batch, 1, hidden_dim*2]
        attn_output, _ = self.attention(query, all_hiddens, all_hiddens)
        # attn_output: [batch, 1, hidden_dim*2]
        aggregated_output = attn_output.squeeze(1)  # [batch, hidden_dim*2]
        return aggregated_output


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input data.
        
        Returns:
            Encoded representation of the input data. [batch size, sequence length, latent dimension], [batch size, sequence length, latent dimension]
        """
        pitch_embeddings = self.pitch_embedding(x[:, :, 0].long())
        linear_out = self.linear(x[:, :, 1:]).relu()
        x = torch.cat((linear_out, pitch_embeddings), dim=-1)
        x = x.contiguous()
        all_hiddens, (h_n, _) = self.lstm(x)
        lstm_out_forward = h_n[-2]
        lstm_out_backward = h_n[-1]
        lstm_out = torch.cat((lstm_out_forward, lstm_out_backward), dim=-1)
        attention = self.attention_aggregate(all_hiddens, lstm_out)
        mu_linear_out = self.mu_linear(attention)
        logvar_linear_out = self.logvar_linear(attention)
        return mu_linear_out, logvar_linear_out
