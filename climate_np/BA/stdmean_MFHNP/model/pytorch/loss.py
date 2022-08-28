import torch
import torch.nn as nn 
from scipy.stats import multivariate_normal
import numpy as np


def nll_loss(pred_mu, pred_cov, y):
        pred_std = torch.sqrt(pred_cov)
        gaussian = torch.distributions.Normal(pred_mu, pred_std)
        nll = -gaussian.log_prob(y)
        nll = torch.mean(nll)
        return nll

def nll_metric(pred_mu, pred_cov, y):
        pred_mu = torch.from_numpy(pred_mu)
        pred_cov = torch.from_numpy(pred_cov)
        y = torch.from_numpy(y)
        pred_std = torch.sqrt(pred_cov)
        gaussian = torch.distributions.Normal(pred_mu, pred_std)
        nll = -gaussian.log_prob(y)
        nll = torch.mean(nll).cpu().detach().numpy()
        return nll

def mae_metric(y_pred, y_true):
    loss = np.abs(y_pred - y_true)
    loss[loss != loss] = 0
    loss = loss.mean()
    return loss

def kld_gaussian_loss(z_mean_all, z_var_all, z_mean_context, z_var_context): 
    """Analytical KLD between 2 Gaussians."""
    mean_q, var_q, mean_p, var_p = z_mean_all, z_var_all, z_mean_context,  z_var_context
    std_q = torch.sqrt(var_q)
    std_p = torch.sqrt(var_p)
    p = torch.distributions.Normal(mean_p, std_p)
    q = torch.distributions.Normal(mean_q, std_q)
    return torch.mean(torch.sum(torch.distributions.kl_divergence(q, p),dim=1))

