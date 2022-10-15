import copy
import torch
import torch.nn as nn

def l2norm(X):
    """
    L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X