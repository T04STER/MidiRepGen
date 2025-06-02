from torch import nn
import torch

from src.models.representation.vae.decoder.vae_decoder import VaeDecoder
from src.models.representation.vae.encoder.vae_encoder import VaeEncoder




class RecurrentVae(nn.Module):
    def __init__(self, latent_dim, encoder: VaeEncoder, decoder: VaeDecoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar
    
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def sample(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Sample from the VAE using the provided noise tensor.
        
        Args:
            noise (torch.Tensor): Noise tensor to sample from.
        
        Returns:
            torch.Tensor: Sampled output from the decoder.
        """
        return self.decoder(noise)
    
