import numpy as np
import torch
from utils import device
from deepinv.physics import LinearPhysics, BlurFFT, GaussianNoise, Decolorize
from deepinv.physics.blur import gaussian_blur
from torchvision import transforms
from deepinv.utils.demo import load_dataset
from torchvision.datasets import ImageFolder


def generate_measurements_gaussian_diag(d, sigmax, sigma):
    d = int(d)
    x = torch.tensor(sigmax*np.random.normal(size=d)).to(device).reshape((1, 1, d))
    
    p = LinearPhysics(  # identity, for compatibility
        img_size=(1, d),
        device=device)
    
    y = p(x) + torch.tensor(sigma*np.random.normal(size=d)).to(device)
    return y, x, p


def generate_measurements_laplace(img_size, sigmax, sigma, sigma_blur=0.1, dtype=np.float32):
    # apply a gaussian blur
    x = torch.tensor(np.random.laplace(0., sigmax, [1, 1, img_size, img_size]).astype(dtype)).to(device)
    p = generate_blur_operator(img_size, sigma_blur=sigma_blur, sigma=sigma, dtype=dtype)
    y = p(x)    
    return y, x, p
   

def generate_gaussian_blur_operator(img_size, sigma, sigma_blur=0.1, dtype=torch.float32):
    filter_torch = gaussian_blur(sigma=(sigma_blur, sigma_blur)).to(device)
 
    return generate_blur_operator(img_size, filter_torch, sigma)


def generate_blur_operator(img_size, filter_torch, sigma):
    return BlurFFT(img_size=(1, img_size, img_size), filter=filter_torch, device=device, padding="circular",
                             noise_model=GaussianNoise(sigma=sigma))
    

def generate_measurements_natural(img_size, sigma, sigma_blur=0.1, im_ind=0):
    dataset_name = "set3c"
    val_transform = transforms.Compose([transforms.CenterCrop(img_size), transforms.ToTensor()])
    dataset = load_dataset("set3c", transform=val_transform)
    x = dataset[im_ind][0].unsqueeze(0).to(device)
    deco = Decolorize(device=device)
    x = deco(x)
    p = generate_gaussian_blur_operator(img_size, sigma_blur=sigma_blur, sigma=sigma)
    y = p(x) 
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
        
