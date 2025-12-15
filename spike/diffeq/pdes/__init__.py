"""
SPIKE PDEs

Available 1D PDEs:
- BurgersEquation: Viscous Burgers (u_t + u*u_x = nu*u_xx)
- HeatEquation: Heat/diffusion (u_t = alpha*u_xx)
- AdvectionEquation: Linear advection (u_t + c*u_x = 0)
- WaveEquation: Wave (u_tt = c^2*u_xx)
- KdVEquation: Korteweg-de Vries (u_t + u*u_x + u_xxx = 0)
- AllenCahnEquation: Phase field (u_t = eps^2*u_xx + u - u^3)
- CahnHilliardEquation: Phase separation (4th order)
- KuramotoSivashinskyEquation: Chaotic PDE (u_t + u*u_x + u_xx + u_xxxx = 0)
- ReactionDiffusionEquation: Reaction-diffusion (u_t = D*u_xx + R(u))
- SchrodingerEquation: Nonlinear Schrodinger (i*u_t + u_xx + |u|^2*u = 0)

Available 2D PDEs:
- Heat2D: 2D Heat/diffusion (u_t = alpha*(u_xx + u_yy))
- Wave2D: 2D Wave (u_tt = c^2*(u_xx + u_yy))
- Burgers2D: 2D Burgers convection-diffusion
- NavierStokes2D: Incompressible Navier-Stokes
- NavierStokes2DLidDriven: Classic CFD benchmark (lid-driven cavity)
- NavierStokes2DChannel: Poiseuille flow (channel flow)
"""

from .burgers import BurgersEquation
from .heat import HeatEquation
from .advection import AdvectionEquation
from .wave import WaveEquation
from .kdv import KdVEquation
from .allen_cahn import AllenCahnEquation
from .cahn_hilliard import CahnHilliardEquation
from .kuramoto_sivashinsky import KuramotoSivashinskyEquation
from .reaction_diffusion import ReactionDiffusionEquation
from .schrodinger import SchrodingerEquation
from .heat_2d import Heat2D
from .wave_2d import Wave2D
from .burgers_2d import Burgers2D
from .navier_stokes_2d import NavierStokes2D, NavierStokes2DLidDriven, NavierStokes2DChannel

__all__ = [
    # 1D PDEs
    'BurgersEquation',
    'HeatEquation',
    'AdvectionEquation',
    'WaveEquation',
    'KdVEquation',
    'AllenCahnEquation',
    'CahnHilliardEquation',
    'KuramotoSivashinskyEquation',
    'ReactionDiffusionEquation',
    'SchrodingerEquation',
    # 2D PDEs
    'Heat2D',
    'Wave2D',
    'Burgers2D',
    'NavierStokes2D',
    'NavierStokes2DLidDriven',
    'NavierStokes2DChannel',
]
