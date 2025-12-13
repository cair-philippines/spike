"""
UDE: Universal Differential Equations
PINN encoder + UnknownNN for learning missing dynamics.

    u_t = f_known(u, u_x, ...) + NN(u, u_x, ...)

Rackauckas et al. (2020). Universal Differential Equations for Scientific Machine Learning.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .pinn import PINN
from .unknown_nn import UnknownNN


class UDE(nn.Module):
    """
    Universal Differential Equation model.

    Architecture:
        1. PINN: (x, t) → u(x, t)
        2. UnknownNN: (u, u_x, ...) → f_unknown

    Same architecture as PIKE for fair comparison.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        unknown_features: Optional[List[str]] = None,
        unknown_hidden: int = 64,
        unknown_layers: int = 2,
        activation: str = 'tanh'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.unknown_features = unknown_features or ['u']

        # PINN encoder
        self.pinn = PINN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation
        )

        # Unknown dynamics NN
        self.unknown_nn = UnknownNN(
            input_features=self.unknown_features,
            hidden_dim=unknown_hidden,
            num_layers=unknown_layers,
            output_dim=output_dim,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (x, t) → u."""
        return self.pinn(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.pinn(x)

    def get_unknown_term(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        derivatives: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute learned unknown term NN(u, u_x, ...)."""
        if derivatives is None:
            derivatives = self.compute_derivatives(u, x)
        features = self._build_features(u, derivatives)
        return self.unknown_nn(features)

    def compute_derivatives(
        self,
        u: torch.Tensor,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute derivatives of u w.r.t. x."""
        derivatives = {}
        is_2d = self.input_dim == 3

        # First derivatives
        grad_u = torch.autograd.grad(
            u, x, torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        derivatives['u_x'] = grad_u[:, 0:1]
        if is_2d:
            derivatives['u_y'] = grad_u[:, 1:2]
        t_idx = 2 if is_2d else 1
        derivatives['u_t'] = grad_u[:, t_idx:t_idx+1]

        # Second derivatives if needed
        if any(f in self.unknown_features for f in ['u_xx', 'u_xxx']):
            u_xx = torch.autograd.grad(
                derivatives['u_x'], x, torch.ones_like(derivatives['u_x']),
                create_graph=True, retain_graph=True
            )[0][:, 0:1]
            derivatives['u_xx'] = u_xx

        if is_2d and 'u_yy' in self.unknown_features:
            u_yy = torch.autograd.grad(
                derivatives['u_y'], x, torch.ones_like(derivatives['u_y']),
                create_graph=True, retain_graph=True
            )[0][:, 1:2]
            derivatives['u_yy'] = u_yy

        # Third derivatives if needed
        if 'u_xxx' in self.unknown_features:
            u_xxx = torch.autograd.grad(
                derivatives['u_xx'], x, torch.ones_like(derivatives['u_xx']),
                create_graph=True, retain_graph=True
            )[0][:, 0:1]
            derivatives['u_xxx'] = u_xxx

        return derivatives

    def _build_features(
        self,
        u: torch.Tensor,
        derivatives: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Build features dict for unknown NN."""
        features = {}
        for f in self.unknown_features:
            if f == 'u':
                features['u'] = u
            elif f == 'u2':
                features['u2'] = u ** 2
            elif f == 'u3':
                features['u3'] = u ** 3
            elif f == 'u_u3':
                features['u_u3'] = u - u ** 3
            elif f == 'u_ux':
                features['u_ux'] = u * derivatives['u_x']
            elif f in derivatives:
                features[f] = derivatives[f]
        return features

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
