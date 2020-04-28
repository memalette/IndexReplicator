import torch
import torch.nn as nn


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def init_weights(m):
    if isinstance(m, nn.Linear):
      nn.init.normal_(m.weight, mean=0., std=0.1)
      nn.init.constant_(m.bias, 0.1)


class Exp(nn.Module):

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        out = torch.exp(x)
        out = torch.clamp(out, min=0.0001)
        return out