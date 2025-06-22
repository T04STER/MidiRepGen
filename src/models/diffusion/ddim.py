import torch


from .ddpm import DDPM

class DDIM(DDPM):
    def q_posterior(self, x_t, x_0, t, noise=None):
        
        alpha_bar_prev_t = self.alpha_bars_prev[t]
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev_t)

        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1 - alpha_bar_prev_t)

        # Ekstrakcja szumu epsilon_t ze wzoru reparametryzacji
        eps = self._predict_eps_from_xstart(x_t, t, x_0)

        K = x_t.dim() - 1
        sqrt_alpha_bar_prev = sqrt_alpha_bar_prev.view(-1, *([1]*K)).expand_as(x_t)
        sqrt_one_minus_alpha_bar_prev = sqrt_one_minus_alpha_bar_prev.view(-1, *([1]*K)).expand_as(x_t)

        x_prev = sqrt_alpha_bar_prev * x_0 + sqrt_one_minus_alpha_bar_prev * eps

        return x_prev