import numpy as np
import torch
from utils import device
import deepinv as dinv


def generate_measurements_gaussian_diag(d, sigmax, sigma):
    d = int(d)
    x = torch.tensor(sigmax*np.random.normal(size=d)).to(device).reshape((1, 1, d))
    
    p = dinv.physics.LinearPhysics(  # identity, for compatibility
        img_size=(1, d),
        device=device)
    
    y = p(x) + torch.tensor(sigma*np.random.normal(size=d)).to(device)
    return y, x, p

def compute_evidence_gaussian_diag(d, sigmax, sigma, y, mlog=True):  # sum of X and a gaussian of variance sigma^2, -log () if mlog is True
    res = 0.5*np.sum(y**2)/(sigmax**2 + sigma**2)
    res += 0.5*d*np.log(2*np.pi)  # normalization of likelihood term

    if mlog == False:
        res = np.exp(-res)
    return res


def compute_test_gaussian_diag(d, sigmax, sigma, alpha, y, yp, ym, mlog=True):  # p(y+/y-) 
    sigmax2, sigma2 = sigmax**2, sigma**2 
    res = -0.5*np.sum(y**2)*sigmax2/sigma2/(sigmax2 + sigma2) + 0.5 * alpha**2 * sigmax2 * np.sum(ym**2) / sigma2 / (alpha*sigmax2 + sigma2)
    res += 0.5 * (1-alpha) * np.sum(yp**2) / sigma2
    res += d*np.log(sigma) + 0.5*d*np.log(sigma2 + sigmax2)
    res += 0.5*d*np.log(2*np.pi)
    
    res -= 0.5*d*np.log(1-alpha)
    res -= 0.5*d*np.log(alpha*sigmax2 + sigma2)

    if mlog == False:
        return np.exp(-res)
    else:
        return res
        
