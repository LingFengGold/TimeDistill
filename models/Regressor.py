import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs, input_dim, output_dim, num_layers=2):
        super(Model, self).__init__()
        self.configs = configs

        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.ReLU())
        for i in range(1, num_layers - 2):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(output_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)