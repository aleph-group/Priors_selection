import numpy as np
import torch
import json 

device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    