import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.SiLU())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x):
        return self.net(x)
