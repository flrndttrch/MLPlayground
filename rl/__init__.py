import torch

def use_gpu():
    return torch.cuda.is_available()