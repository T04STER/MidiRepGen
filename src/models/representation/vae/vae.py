from torch import nn
import torch

from src.models.representation.vae.decoder.vae_decoder import VaeDecoder
from src.models.representation.vae.encoder.vae_encoder import VaeEncoder




class RecurrentVae(nn.Module):
    def __init__(self, encoder: VaeEncoder, decoder: VaeDecoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = x.contiguous()
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        x_reconstructed = self.decoder(z, seq_length=x.size(1))
        return x_reconstructed, mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    @torch.inference_mode()
    def sample(self, noise: torch.Tensor, seq_length) -> torch.Tensor:
        """
        Sample from the VAE using the provided noise tensor.
        
        Args:
            noise (torch.Tensor): Noise tensor to sample from.
        
        Returns:
            torch.Tensor: Sampled output from the decoder.
        """
        return self.decoder(noise)


class RecurrentVaeWithTeacherForcing(RecurrentVae):
    def __init__(self, encoder: VaeEncoder, decoder: VaeDecoder, *args, **kwargs):
        super().__init__(encoder, decoder, *args, **kwargs)
    
        
    def forward(self, x):
        x = x.contiguous()
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        x_reconstructed = self.decoder(z, seq_length=x.size(1), true_output=x)
        # x_reconstructed = self.decoder(mu, seq_length=x.size(1), true_output=x)
        return x_reconstructed, mu, logvar
    





    