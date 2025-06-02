from torch import nn
import torch


class VaeLoss(nn.Module):
    def __init__(self):
        super(VaeLoss, self).__init__()

        self.mse_loss = nn.MSELoss(reduction="sum")  # not sure yet

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(
            mu * mu - logvar.exp() - logvar -1,
            dim=1,
        ).sum() # not sure yet

    def forward(
        self,
        x_reconstructed: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        divergence = self.kl_divergence(mu, logvar)
        reconstruction_loss = self.mse_loss(x_reconstructed, x)
        return reconstruction_loss + divergence
