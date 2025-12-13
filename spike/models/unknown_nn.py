"""
Unknown Terms NN
MLP that learns unknown/missing dynamics in differential equations.

Part of UDE framework (Rackauckas et al., 2020).
"""

import torch
import torch.nn as nn
from typing import List, Dict


class UnknownNN(nn.Module):
    """
    Neural network for learning unknown terms: (u, u_x, ...) → f_unknown.

    Initialized with small weights so output starts near zero.
    """

    def __init__(
        self,
        input_features: List[str] = ['u'],
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        activation: str = 'tanh'
    ):
        super().__init__()
        self.input_features = input_features
        self.input_dim = len(input_features)
        self.output_dim = output_dim

        # Activation
        if activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'relu':
            act = nn.ReLU()
        elif activation == 'silu':
            act = nn.SiLU()
        else:
            act = nn.Tanh()

        # Build MLP
        layers = [nn.Linear(self.input_dim, hidden_dim), act]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Small init so NN starts near zero
        self._init_small()

    def _init_small(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward: features dict → unknown term."""
        inputs = torch.cat([features[f] for f in self.input_features], dim=-1)
        return self.net(inputs)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
