from torch import nn
import torch


class VaeLoss(nn.Module):
    def __init__(self, reduction="sum", beta=1.0):
        super(VaeLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def mse_loss(self, x_reconstructed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            batch_size = x.size(0)
            return 1/batch_size * (x - x_reconstructed).pow_(2).sum()
        elif self.reduction == "sum":
            return nn.functional.mse_loss(x_reconstructed, x, reduction='sum')
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")

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


class VaeLossWithCrossEntropy(nn.Module):
    def __init__(self, reduction="sum", beta=1.0):
        super(VaeLossWithCrossEntropy, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    def cross_entropy_loss(self, x_reconstructed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            batch_size = x.size(0)
            # x is expected to be of shape [batch_size, sequence_length, num_classes]
            # x_reconstructed is expected to be of shape [batch_size, sequence_length]
            x_reconstructed = x_reconstructed.view(-1, 128)
            x = x.view(-1)
            return self.ce_loss_fn(x_reconstructed, x).sum()/batch_size
        elif self.reduction == "sum":
            return self.ce_loss_fn(x_reconstructed, x).sum()
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")

    def mse_loss(self, x_reconstructed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            batch_size = x.size(0)
            return(x - x_reconstructed).pow_(2).sum()/batch_size
        elif self.reduction == "sum":
            return nn.functional.mse_loss(x_reconstructed, x, reduction='sum')
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")

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
    ):
        pitch_logits, others_to_mse = x_reconstructed
        divergence = self.beta*self.kl_divergence(mu, logvar)
        pitch_targets = x[:, :, 0]
        # print(f"Pitch one-hot shape: {pitch_one_hot.shape}, Pitch targets shape: {pitch_targets.shape}")
        ce_loss = self.cross_entropy_loss(pitch_logits, pitch_targets.long())
        # print(f"CE Loss: {ce_loss.item()}, Divergence: {divergence.item()}")
        mse_loss = self.mse_loss(others_to_mse, x[:, :, 1:])
        # print(f"MSE Loss: {mse_loss.item()}")
        return ce_loss + mse_loss + divergence, (ce_loss.item(), mse_loss.item(), divergence.item())
