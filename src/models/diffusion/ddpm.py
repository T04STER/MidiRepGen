import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def _apply(coeficients: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    factors = coeficients[timesteps.long()].float()
    K = x.dim() - 1
    factors = factors.view(-1, *([1] * K)).expand(x.shape)
    return x * factors


class DDPM:
    def __init__(self, num_timesteps: int=1_000, device: str='cuda'):
        self.device = device
        self.num_timesteps = num_timesteps
        self.default_num_timesteps = num_timesteps
        self.timesteps = torch.arange(0, num_timesteps, dtype=torch.float32, device=device)
        self.setup_noise_scheduler()
    
    def setup_noise_scheduler(self, betas:torch.Tensor|None=None):
        self.betas = torch.linspace(1e-4, 0.02, self.num_timesteps).to(self.device) if betas is None else betas
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat((torch.tensor([1.0]).to(device=self.device), self.alpha_bars[:-1]))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t_mu = _apply(self.alpha_bars, t, x_start)
        x_t_sigma = _apply(self.betas.sqrt(), t, torch.ones_like(x_start))
        x_t_sigma = _apply(torch.sqrt(1.0 - self.alpha_bars), t, noise)
        x_t = x_t_mu + x_t_sigma
        return x_t
    
    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        ones = torch.ones_like(x_t)
        alpha_sqrt = _apply(self.alphas, t, ones).sqrt()
        alpha_bar_prev_sqrt = _apply(self.alpha_bars_prev, t, ones).sqrt()
        betas = _apply(self.betas, t, ones)
        _1_prev_alpha_bar = 1 - _apply(self.alpha_bars_prev, t, ones)
        _1_alpha_bar = 1- _apply(self.alpha_bars, t, ones)

        numerator = (
            (alpha_sqrt * _1_prev_alpha_bar * x_t ) 
            + (alpha_bar_prev_sqrt * betas * x_start)
        )
        mu = numerator / (_1_alpha_bar + 10e-8)
        sigma = torch.sqrt(
            (betas * _1_prev_alpha_bar) / (_1_alpha_bar + 10e-8)
        )

        return mu + sigma * noise
    
    def _predict_x_0_from_eps(self, x_t, t, eps):
        return (
            _apply(torch.sqrt(1.0 / self.alpha_bars), t, x_t) - 
            _apply(torch.sqrt(1.0 / self.alpha_bars - 1.0), t, eps)
        )

    def _predict_eps_from_xstart(self, x_t, t, x_0):
        return (
            _apply(torch.sqrt(1.0 / self.alpha_bars), t, x_t) - x_0
        ) / _apply(torch.sqrt(1.0 / self.alpha_bars - 1.0), t, torch.ones_like(x_t))

    def train_losses(self, model, x_0):
        loss = 0
        rand_t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device).long()
        eps = torch.randn_like(x_0).to(x_0.device)
        x_t = self.q_sample(x_0, rand_t, noise=eps)
        eps_pred = model(x_t, rand_t)
        loss = F.mse_loss(eps_pred, eps)
        return loss
    
    @torch.inference_mode
    def p_sample_loop(
        self, 
        model: nn.Module, 
        noise: torch.Tensor, 
        num_inference_steps: int = 1000,
        return_trajectory: bool = False, 
        clip: bool = False,
        quiet: bool = True
        ):
        self._respace(num_timesteps=num_inference_steps)

        x_t = noise
        bsz = x_t.shape[0]
        trajectory = [x_t.clone().cpu()]

        pbar = tqdm(enumerate(self.timesteps), desc='Sampling', total=self.num_timesteps) if not quiet else enumerate(self.timesteps)

        for idx, time in pbar:
            t = torch.tensor([time] * bsz, device=x_t.device).long()
            i = torch.tensor([self.num_timesteps - idx - 1] * bsz, device=x_t.device).long()
            eps = model(x_t, t)
            x_0 = self._predict_x_0_from_eps(x_t, i, eps)
            if clip:
                x_0 = x_0.clamp(-1, 1)
            x_t = self.q_posterior(x_t, x_0, i)

        self._respace(self.default_num_timesteps)

        if return_trajectory:
            return x_0.cpu().numpy(), torch.stack(trajectory, dim=0).numpy()
        return x_0.cpu().numpy()

    def _respace(self, num_timesteps):
        self.setup_noise_scheduler() 

        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(999, 0, self.num_timesteps, dtype=torch.float32, ).long()

        last_alpha_cumprod = 1.0

        betas = []

        for i, alpha_bar in enumerate(self.alpha_bars):
            if i in self.timesteps:
                betas.append(1 - alpha_bar / last_alpha_cumprod)
                last_alpha_cumprod = alpha_bar
        
        self.betas = torch.tensor(betas, dtype=torch.float32).to(self.device)
        self.setup_noise_scheduler(self.betas)