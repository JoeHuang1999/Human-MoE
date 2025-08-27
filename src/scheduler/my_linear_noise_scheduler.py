import torch
import numpy as np


class LinearNoiseScheduler:
    """
    Class for the linear noise scheduler that is used in DDPM.
    """
    
    def __init__(self, num_timesteps, beta_start, beta_end):
        """
        self.num_timesteps = 1000
        self.beta_start = 0.00085
        self.beta_end = 0.012
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        """
        mimicking how compvis repo creates schedule
        torch.linspace(start=-10, end=10, steps=6):
        tensor([-10., -6., -2., 2., 6., 10.])
        self.betas: tensor([0.0008, ..., 0.0120])
        """
        self.betas = (
            torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        )

        # self.alphas: tensor([0.9991, ..., 0.9880])
        self.alphas = 1. - self.betas

        """
        torch.cumprod(torch.tensor([1, 2, 3, 4, 5]), dim=0):
        tensor([1, 2, 6, 24, 120])
        self.alpha_cum_prod: tensor([0.9991, ..., 0.0047])
        """
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        """
        torch.sqrt(torch.tensor([4, 9, 16])):
        tensor([2., 3., 4.])
        self.sqrt_alpha_cum_prod: 
        self.alpha_cum_prod: tensor([0.9996, ..., 0.0683])
        """
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)

        """
        self.sqrt_one_minus_alpha_cum_prod: tensor([0.0292, ..., 0.9977])
        """
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """

        # torch.Size([16, 3, 32, 32])
        original_shape = original.shape
        # 16
        batch_size = original_shape[0]
        # shape: torch.Size([16])
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        # shape: torch.Size([16])
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        # (B,) => (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            # shape: torch.Size([16, 1, 1, 1])
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            # shape: torch.Size([16, 1, 1, 1])
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        # apply and return forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def add_one_step_noise(self, original, noise, t, step_size=1):
        # torch.Size([16, 3, 32, 32])
        original_shape = original.shape
        # 16
        batch_size = original_shape[0]
        # shape: torch.Size([16])
        sqrt_one_sub_beta_t_pre = torch.sqrt(1 - self.betas.to(original.device)[t]).reshape(batch_size)
        sqrt_beta_t_cur = torch.sqrt(self.betas.to(original.device)[t+step_size]).reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            # shape: torch.Size([16, 1, 1, 1])
            sqrt_one_sub_beta_t_pre = sqrt_one_sub_beta_t_pre.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            # shape: torch.Size([16, 1, 1, 1])
            sqrt_beta_t_cur = sqrt_beta_t_cur.unsqueeze(-1)
        # apply and return forward process equation
        return (sqrt_one_sub_beta_t_pre.to(original.device) * original
                + sqrt_beta_t_cur.to(original.device) * noise)
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the nosie predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0

    def sample_prev_timestep_ddim(self, xt, noise_pred, t, ddim_step=1, eta=1):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the nosie predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)
        ab_cur = self.alpha_cum_prod.to(xt.device)[t]
        ab_prev = self.alpha_cum_prod.to(xt.device)[t - ddim_step] if t - ddim_step >= 0 else 1.
        variance = eta * (1. - ab_prev) / (1. - ab_cur) * (1. - ab_cur / ab_prev)
        z = torch.randn(xt.shape).to(xt.device)
        term_1 = (ab_prev / ab_cur) ** 0.5 * xt
        term_2 = (torch.sqrt(1 - ab_prev - variance) - torch.sqrt(ab_prev * (1. - ab_cur) / ab_cur)) * noise_pred
        term_3 = variance**0.5 * z
        output = term_1 + term_2 + term_3
        return output, x0