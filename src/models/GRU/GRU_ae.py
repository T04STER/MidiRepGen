import torch.nn as nn

class GRUAutoencoder(nn.Module):
    def __init__(self, pitch_dim=84, embed_dim=128, hidden_dim=256, latent_dim=64):
        super().__init__()

        self.pitch_embedding = nn.Embedding(pitch_dim, embed_dim)
        self.velocity_proj = nn.Linear(1, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)
        self.duration_proj = nn.Linear(1, embed_dim)


        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.out_pitch = nn.Linear(hidden_dim, pitch_dim)
        self.out_velocity = nn.Linear(hidden_dim, 1)
        self.out_time = nn.Linear(hidden_dim, 1)
        self.out_duration = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        pitch, velocity, time, duration = x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3]

        emb = (
            self.pitch_embedding(pitch.long())
            + self.velocity_proj(velocity.unsqueeze(-1))
            + self.time_proj(time.unsqueeze(-1))
            + self.duration_proj(duration.unsqueeze(-1))
        )
        
        _, h_n = self.encoder_gru(emb)
        latent = self.latent_proj(h_n[-1])
        
        h_0 = self.latent_to_hidden(latent).unsqueeze(0)
        output, _ = self.decoder_gru(emb, h_0)

        pitch_logits = self.out_pitch(output)
        velocity_logits = self.out_velocity(output)
        time_logits = self.out_time(output)
        duration_logits = self.out_duration(output)

        return {
            'pitch': pitch_logits,
            'velocity': velocity_logits,
            'time': time_logits,
            'duration': duration_logits
        }

