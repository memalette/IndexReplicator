import torch
import torch.nn as nn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def init_weights(m):
    if isinstance(m, nn.Linear):
      nn.init.normal_(m.weight, mean=0., std=0.1)
      nn.init.constant_(m.bias, 0.1)


def hyperparam_search(f, space, max_trials):

    trials = Trials()
    best = fmin(f, space, algo=tpe.suggest, max_evals=max_trials, trials=trials)
    print('Best: ', best)

    print('Trials:')
    for trial in trials.trials:
        print(trial)

    return best


class Exp(nn.Module):

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        out = torch.exp(x)
        out = torch.clamp(out, min=0.0001)
        return out
