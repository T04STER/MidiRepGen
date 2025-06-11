from torch import nn
import torch


class VaeLoss(nn.Module):
    def __init__(self, reduction="sum", beta=1.0):
        super(VaeLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)  # not sure yet

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        else:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl
    
    def forward(
        self,
        x_reconstructed: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        divergence = self.beta*self.kl_divergence(mu, logvar)
        reconstruction_loss = self.mse_loss(x_reconstructed, x)
        return reconstruction_loss + divergence


