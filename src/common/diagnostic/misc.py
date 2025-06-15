


import typing
import numpy as np
import torch

from src.common.diagnostic.visualize_note_events import compare_reco_true
from src.dataloader.dataset import EventMidiDataset
from src.models.representation.vae.vae import RecurrentVae
from torch.utils.data import DataLoader


# type: ignore
@torch.inference_mode
def vae_adapter(x, vae: RecurrentVae) -> torch.Tensor:
    """
        A simple adapter to use the VAE with a single input tensor. Wraps model to provide proper output shape
        as RecurrentVAE outputs one-hot encoded pitches and other features concatenated.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, 4].
            vae (RecurrentVae): The VAE model to use for reconstruction.
        Returns:
            torch.Tensor: The reconstructed tensor of shape [batch_size, seq_length, 4].
    """
    vae.eval()

    out, _, _ = vae(x.cuda())
    pitches_one_hot, others = out
    pitches = (torch.argmax(pitches_one_hot, dim=-1)).unsqueeze(-1)
    reconstruction = torch.cat([pitches, others], dim=-1)
    return reconstruction


@torch.inference_mode # type: ignore
def get_random_reconstruction_true_pair_from_note_event(vae: RecurrentVae, dataloader: DataLoader[EventMidiDataset] | EventMidiDataset, strip_to_seq_length=None) -> tuple[torch.Tensor, np.ndarray]:
    """
        Get a random sample from the dataset and its reconstruction using the VAE.
    """
    if isinstance(dataloader, EventMidiDataset):
        dataset = dataloader
    else:
        dataset: EventMidiDataset = dataloader.dataset
    idx = np.random.randint(0, len(dataset), size=2)
        
    first_sample = dataset[idx[0]].unsqueeze(0)
    reconstructed_sample = vae_adapter(first_sample, vae).squeeze(0)  # type: ignore
    if strip_to_seq_length is not None:
        first_sample = first_sample[:strip_to_seq_length, :]
        reconstructed_sample = reconstructed_sample[:strip_to_seq_length, :]
    
    return first_sample, reconstructed_sample.detach().cpu().numpy()



def plot_random_reconstruction_true_pair(vae: RecurrentVae, dataloader: DataLoader[EventMidiDataset] | EventMidiDataset, strip_to_seq_length=None):
    """
        Plot a random sample from the dataset and its reconstruction using the VAE.
    """
    randoms = get_random_reconstruction_true_pair_from_note_event(vae, dataloader, strip_to_seq_length)
    compare_reco_true(*randoms)
