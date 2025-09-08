import arviz 
import numpy as np
import torch
import json 

device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def plot_MC_corr(X_trace, ax, title="X"):
    """ 
    Plot the autocorrelation of the MC X_trace if shape (n_samples, 1), and plot the autocorrelation of the min, median and
    max variance pixels if shape (n_samples, n_pixels). Displays the effective sampling size in plot title.
    """
    if X_trace.shape[1] > 1:  # shape (n_samples, n_pixels)
        X_var = np.var(X_trace, axis=0)   
        # track pixels with max, min, med variances
        sort_ind = np.argsort(X_var)
        X_max_var = X_trace[:, sort_ind[-1]]
        X_min_var = X_trace[:, sort_ind[0]]
        X_med_var = X_trace[:, sort_ind[len(sort_ind) // 2]]
        ds = dict(Xmax=X_max_var, Xmin=X_min_var, Xmed=X_med_var)
        # corresponding effective sample sizes
        ess_max = arviz.ess(X_max_var.reshape(-1))
        ess_min = arviz.ess(X_min_var.reshape(-1))
        ess_med  = arviz.ess(X_med_var.reshape(-1))
        arviz.plot_autocorr(ds, var_names=['Xmin', 'Xmed', 'Xmax'], ax=ax)
        ax[0].set_title(title + " min: {:.1f}".format(ess_min))
        ax[1].set_title(title + " med: {:.1f}".format(ess_med))
        ax[2].set_title(title + " max: {:.1f}".format(ess_max))
    else:  # shape (n_samples,)
        ess = arviz.ess(X_trace.reshape(-1))
        ds = dict(X=X_trace.reshape(-1))
        arviz.plot_autocorr(ds, var_names=['X'], ax=ax)
        ax[0].set_title(title + ": {:.1f}".format(ess))


def tcheby(x, n):
    if x >= 1:
        return torch.cosh(n * torch.acosh(x))
    if x <= -1:
        return torch.cosh(n * torch.acosh(-x))*(-1)**n
    else:
        return torch.cos(n * torch.acos(x))
    

def tcheby_der(x, n):
    if x >= 1:
        return n * torch.sinh(n * torch.acosh(x)) / torch.sqrt(x**2 - 1)
    if x <= -1:
        return n * torch.sinh(n * torch.acosh(-x)) / torch.sqrt(x**2 - 1) * (-1)**(n+1)
    else:
        return n * torch.sin(n * torch.acos(x)) / torch.sqrt(1 - x**2)

def normalize(x):
    "Project x in the [0, 1] range."
    flatx = x.view(x.shape[0], -1)
    minval, maxval = torch.min(flatx, dim=1).values, torch.max(flatx, dim=1).values
    return (x - minval[:, None, None, None])/(maxval - minval)[:, None, None, None]


def moffat(alpha=(1., 1), size=None):
    if size is None:
        c = int(2*(alpha[1]/alpha[0]**2*(2**(1/(alpha[1]/2+1))-1))**0.5 + 1)
    else:
        c = size
    
    k_size = 2 * c + 1

    delta = torch.arange(k_size)
    x, y = torch.meshgrid(delta, delta, indexing="ij")
    x = x - c
    y = y - c
    filt = alpha[0]**2*(x**2 + y**2)/alpha[1] +  1 
    filt = filt.pow(-(alpha[1]/2 + 1)) * alpha[0]**2 / 2 / torch.pi
    filt = filt / filt.flatten().sum()

    return filt.unsqueeze(0).unsqueeze(0)
    

def laplace(alpha, size=None):
    if size is None:
        c = int(0.2/alpha**2 + 1)#int(3*np.log(2.)/alpha + 1)
    else:
        c = size
    k_size = 2 * c + 1

    delta = torch.arange(k_size)
    x, y = torch.meshgrid(delta, delta, indexing="ij")
    x = x - c
    y = y - c
    filt = torch.exp(-alpha*(torch.abs(x) + torch.abs(y)))*alpha**2/4
    filt = filt / filt.flatten().sum()

    return filt.unsqueeze(0).unsqueeze(0)
    

def uniform(size=5):
    k_size = 2 * size + 1
    filt = torch.ones((k_size, k_size))
    filt = filt / filt.flatten().sum()

    return filt.unsqueeze(0).unsqueeze(0)


def dict_to_json(dico, path):
    dd = dico.copy()
    for key in dd:
        dd[key] =  dd[key].tolist() if type(dd[key]) == np.ndarray else dd[key]
    fd = open(path, 'w')
    json.dump(dd, fd)
    fd.close()


def json_to_dict(path):
    fd = open(path, 'r')
    dd = json.load(fd)
    fd.close()

    for key in dd:
        dd[key] =  np.array(dd[key]) if type(dd[key]) == list else dd[key]
    return dd
    