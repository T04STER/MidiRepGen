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
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input tensor using the encoder.
        
        Args:
            x (torch.Tensor): Input tensor to encode.
        
        Returns:
            torch.Tensor: Encoded representation.
        """
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    @torch.inference_mode()
    def encode_with_reparameterization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor and reparameterize to get the latent representation.
        
        Args:
            x (torch.Tensor): Input tensor to encode.
        
        Returns:
            torch.Tensor: Latent representation after reparameterization.
        """
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    @torch.inference_mode()
    def sample(self, noise: torch.Tensor, seq_length) -> torch.Tensor:
        """
        Sample from the VAE using the provided noise tensor.
        
        Args:
            noise (torch.Tensor): Noise tensor to sample from.
        
        Returns:
            torch.Tensor: Sampled output from the decoder.
        """
        return self.decoder(noise, seq_length=seq_length)


    



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
    

    @torch.inference_mode()
    def sample_formatted(self, noise: torch.Tensor, seq_length) -> torch.Tensor:
        """
        Sample from the VAE using the provided noise tensor and format the output.
        
        Args:
            noise (torch.Tensor): Noise tensor to sample from must match latent dim.
        
        Returns:
            torch.Tensor: Sampled output from the decoder, formatted to format [pitch(0-127), velocity(0-127), start_delta_time(secs), durration(secs)].
        """
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)
        x_reconstructed = self.decoder(noise, seq_length=seq_length)
        pitches_one_hot, others = x_reconstructed
        pitches = (torch.argmax(pitches_one_hot, dim=-1)).unsqueeze(-1)
        generated_sequence = torch.cat([pitches, others], dim=-1)
        return generated_sequence



    