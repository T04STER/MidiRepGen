from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch



def _latent_samples_to_numpy(latent_samples: list[torch.Tensor] | torch.Tensor) -> np.ndarray:
    """
    Converts latent samples to a numpy array.
    Args:
        latent_samples (list[torch.Tensor] | torch.Tensor): List of latent samples or a single tensor of shape
            (samples_num, latent_dim).
    Returns:
        np.ndarray: Numpy array of shape (samples_num, latent_dim).
    """
    if isinstance(latent_samples, torch.Tensor):
        if latent_samples.ndim != 2:
            raise ValueError("Latent samples tensor must be 2D or 1D.")
        return latent_samples.cpu().numpy()
    elif isinstance(latent_samples, list):
        if not all(isinstance(sample, torch.Tensor) for sample in latent_samples):
            raise ValueError("All elements in the list must be torch.Tensor.")
        if not all(sample.ndim == 2 for sample in latent_samples):
            raise ValueError("All latent samples must be 2D tensors.")
        latents_list = [sample.cpu().numpy() for sample in latent_samples]
        return np.concatenate(latents_list, axis=0)
    else:
        raise TypeError("latent_samples must be a list of torch.Tensor or a single torch.Tensor.")

def plot_pca(latent_samples: list[torch.Tensor] | torch.Tensor) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots PCA of the latent space.
    Args:
        latent_samples (list[torch.Tensor] | torch.Tensor): List of latent samples or a single tensor of shape
            (samples_num, latent_dim).
    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of the PCA plot.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(_latent_samples_to_numpy(latent_samples))
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    ax.set_title("PCA of Latent Space")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    return fig, ax


def plot_tsne(latent_samples: list[torch.Tensor] | torch.Tensor) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots t-SNE of the latent space.
    Args:
        latent_samples (list[torch.Tensor] | torch.Tensor): List of latent samples or a single tensor of shape
            (samples_num, latent_dim).
    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of the t-SNE plot.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(_latent_samples_to_numpy(latent_samples))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    ax.set_title("t-SNE of Latent Space")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    
    return fig, ax