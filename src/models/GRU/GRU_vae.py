import torch
import torch.nn as nn
from tqdm import tqdm

class GRUVAE(nn.Module):
    def __init__(self, pitch_dim=84, embed_dim=128, hidden_dim=256, latent_dim=64):
        super().__init__()

        self.pitch_embedding = nn.Embedding(pitch_dim, embed_dim)
        self.velocity_proj = nn.Linear(1, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)
        self.duration_proj = nn.Linear(1, embed_dim)

        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(latent_dim + embed_dim, hidden_dim, batch_first=True)

        self.out_pitch = nn.Linear(hidden_dim, pitch_dim)
        self.out_velocity = nn.Linear(hidden_dim, 1)
        self.out_time = nn.Linear(hidden_dim, 1)
        self.out_duration = nn.Linear(hidden_dim, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        pitch, velocity, time, duration = x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3]

        emb = (
            self.pitch_embedding(pitch.long())
            + self.velocity_proj(velocity.unsqueeze(-1))
            + self.time_proj(time.unsqueeze(-1))
            + self.duration_proj(duration.unsqueeze(-1))
        )

        _, h_n = self.encoder_gru(emb)
        h_n = h_n[-1]

        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)

        h_0 = self.latent_to_hidden(z).unsqueeze(0)

        z_seq = z.unsqueeze(1).repeat(1, emb.size(1), 1)
        dec_input = torch.cat([emb, z_seq], dim=-1)
        output, _ = self.decoder_gru(dec_input, h_0)

        pitch_logits = self.out_pitch(output)
        velocity_logits = self.out_velocity(output)
        time_logits = self.out_time(output)
        duration_logits = self.out_duration(output)

        return {
            'pitch': pitch_logits,
            'velocity': velocity_logits,
            'time': time_logits,
            'duration': duration_logits,
            'mu': mu,
            'logvar': logvar
        }
    
    @torch.no_grad()
    def sample(self, seq_len=32, z=None, device='cpu'):
        if z is None:
            z = torch.randn(1, self.fc_mu.out_features).to(device)

        h = self.latent_to_hidden(z).unsqueeze(0)
        z = z.unsqueeze(0)
        
        pitch = torch.randint(0, self.pitch_embedding.num_embeddings, (1, 1)).to(device)
        velocity = torch.zeros(1, 1, 1).to(device)
        time = torch.zeros(1, 1, 1).to(device)
        duration = torch.zeros(1, 1, 1).to(device)

        outputs = {'pitch': [], 'velocity': [], 'time': [], 'duration': []}

        for _ in tqdm(range(seq_len)):
            emb = (
                self.pitch_embedding(pitch)
                + self.velocity_proj(velocity)
                + self.time_proj(time)
                + self.duration_proj(duration)
            )

            dec_input = torch.cat([emb, z], dim=-1)
            out, h = self.decoder_gru(dec_input, h)

            pitch_logits = self.out_pitch(out)
            velocity_out = self.out_velocity(out)
            time_out = self.out_time(out)
            duration_out = self.out_duration(out)

            # Sample pitch stochastically
            pitch_probs = torch.softmax(pitch_logits.squeeze(1), dim=-1)
            pitch = torch.multinomial(pitch_probs, num_samples=1)

            # Use raw outputs for continuous values
            velocity = velocity_out
            time = time_out
            duration = duration_out

            outputs['pitch'].append(pitch.squeeze().cpu())
            outputs['velocity'].append(velocity.squeeze().cpu())
            outputs['time'].append(time.squeeze().cpu())
            outputs['duration'].append(duration.squeeze().cpu())

        return outputs
