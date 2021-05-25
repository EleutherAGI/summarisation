import torch
import torch.nn as nn
from collections import OrderedDict

class AdapterLayer(nn.Module):
    def __init__(self, input_size, reduction_factor):
        super(AdapterLayer, self).__init__()
        self.skip_adapter = False
        self.adapter = nn.Sequential(nn.Linear(input_size, input_size//reduction_factor),
                                     nn.ReLU(),
                                     nn.Linear(input_size//reduction_factor, input_size))
        self.adapter.apply(self.init_weights)

    def init_weights(self, m, std = 1e-2):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std = std)
            torch.nn.init.normal_(m.bias, std = std)
            m.weight.data = torch.clamp(m.weight.data, min = -2*std, max = 2*std)
            m.bias.data = torch.clamp(m.bias.data, min = -2*std, max = 2*std)
    
    def forward(self, X):
        if self.skip_adapter:
            return X
        else:
            return self.adapter(X) + X


# couldn't get it to work with class inheritance
def add_adapters(model, reduction_factor):
    n_layers = len(model.h)
    hidden_size = model.config.hidden_size
    for n in range(n_layers):
        model.h[n].mlp = nn.Sequential(OrderedDict([('MLP', model.h[n].mlp),
                                                   ('Adapter', AdapterLayer(hidden_size, reduction_factor))]))
    return model

def add_adapter_skip(model):

    def adapter_skip(self, skip):
        n_layers = len(self.h)
        for n in range(n_layers):
            self.h[n].mlp.Adapter.skip_adapter = skip
    model.adapter_skip = adapter_skip.__get__(model)
    return model