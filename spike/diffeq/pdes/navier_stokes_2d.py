"""
2D Navier-Stokes Equations
Incompressible fluid dynamics in 2D
"""

import torch
from ..base import BasePDE


class NavierStokes2D(BasePDE):
    """
    2D Incompressible Navier-Stokes Equations.

    PDEs:
    - ∂u/∂t + u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
    - ∂v/∂t + u·∂v/∂x + v·∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
    - ∂u/∂x + ∂v/∂y = 0 (incompressibility)

    Where:
    - u, v = velocity components
    - p = pressure
    - ν = kinematic viscosity

    Args:
        nu: Kinematic viscosity (default: 0.01)
        domain_x: Spatial domain in x (default: (0, 1))
        domain_y: Spatial domain in y (default: (0, 1))
        domain_t: Time domain (default: (0, 1))
    """

    def __init__(
        self,
        nu: float = 0.01,
        domain_x=(0.0, 1.0),
        domain_y=(0.0, 1.0),
        domain_t=(0.0, 1.0)
    ):
        super().__init__(
            domain_x=domain_x,
            domain_t=domain_t
        )
        self.nu = nu
        self.domain_y = domain_y
        self.output_dim = 3  # u, v, p
        self.input_dim = 3  # x, y, t
        self.name = "NavierStokes2D"

    def get_domain(self):
        """Return full 2D+t domain."""
        return {
            'x_min': self.domain_x[0],
            'x_max': self.domain_x[1],
            'y_min': self.domain_y[0],
            'y_max': self.domain_y[1],
            't_min': self.domain_t[0],
            't_max': self.domain_t[1]
        }

    def residual(self, uvp, inputs):
        """
        Compute Navier-Stokes residuals.

        Args:
            uvp: Output [batch_size, 3] as [u, v, p]
            inputs: Input [batch_size, 3] as [x, y, t], requires_grad=True

        Returns:
            Residual [batch_size, 3] for momentum_x, momentum_y, continuity
        """
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]

        # Compute gradients w.r.t. full inputs tensor, then slice
        # First derivatives of u
        grad_u = torch.autograd.grad(
            u, inputs, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]
        u_t = grad_u[:, 2:3]

        # First derivatives of v
        grad_v = torch.autograd.grad(
            v, inputs, grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )[0]
        v_x = grad_v[:, 0:1]
        v_y = grad_v[:, 1:2]
        v_t = grad_v[:, 2:3]

        # First derivatives of p
        grad_p = torch.autograd.grad(
            p, inputs, grad_outputs=torch.ones_like(p),
            create_graph=True, retain_graph=True
        )[0]
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]

        # Second derivatives of u
        grad_u_x = torch.autograd.grad(
            u_x, inputs, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        u_xx = grad_u_x[:, 0:1]

        grad_u_y = torch.autograd.grad(
            u_y, inputs, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0]
        u_yy = grad_u_y[:, 1:2]

        # Second derivatives of v
        grad_v_x = torch.autograd.grad(
            v_x, inputs, grad_outputs=torch.ones_like(v_x),
            create_graph=True, retain_graph=True
        )[0]
        v_xx = grad_v_x[:, 0:1]

        grad_v_y = torch.autograd.grad(
            v_y, inputs, grad_outputs=torch.ones_like(v_y),
            create_graph=True, retain_graph=True
        )[0]
        v_yy = grad_v_y[:, 1:2]

        # Momentum equations
        res_u = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        res_v = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        # Continuity equation (incompressibility)
        res_cont = u_x + v_y

        return torch.cat([res_u, res_v, res_cont], dim=1)

    def get_params(self):
        return {
            'nu': self.nu,
            'Re': 1.0 / self.nu if self.nu > 0 else float('inf')
        }

    def initial_condition(self, x):
        """Default initial condition: zero velocity field."""
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
        return torch.zeros(batch_size, 3)  # u=0, v=0, p=0

    def boundary_condition(self, points, boundary='left'):
        """
        Boundary conditions for channel flow.

        Args:
            points: Boundary points [batch_size, 3] as (x, y, t)
            boundary: Which boundary ('left', 'right', 'top', 'bottom')

        Returns:
            torch.Tensor: Boundary values [batch_size, 3] as (u, v, p)
        """
        batch_size = points.shape[0] if isinstance(points, torch.Tensor) else 1
        device = points.device if isinstance(points, torch.Tensor) else 'cpu'

        if boundary == 'left':  # Inlet
            u = torch.ones(batch_size, 1, device=device)
            v = torch.zeros(batch_size, 1, device=device)
            p = torch.zeros(batch_size, 1, device=device)
            return torch.cat([u, v, p], dim=1)

        elif boundary == 'right':  # Outlet - match inlet for mass conservation
            u = torch.ones(batch_size, 1, device=device)
            v = torch.zeros(batch_size, 1, device=device)
            p = torch.zeros(batch_size, 1, device=device)
            return torch.cat([u, v, p], dim=1)

        elif boundary in ['top', 'bottom']:  # No-slip walls
            return torch.zeros(batch_size, 3, device=device)

        # Default: no-slip
        return torch.zeros(batch_size, 3, device=device)

    def get_reynolds_number(self, L: float = 1.0, U: float = 1.0) -> float:
        """Compute Reynolds number Re = UL/ν."""
        return U * L / self.nu


class NavierStokes2DLidDriven(NavierStokes2D):
    """
    Lid-Driven Cavity Flow (classic CFD benchmark).

    Boundary conditions:
    - Top: u=1, v=0 (moving lid)
    - Other walls: u=0, v=0 (no-slip)
    """

    def __init__(self, nu: float = 0.01, domain_size: float = 1.0):
        super().__init__(
            nu=nu,
            domain_x=(0.0, domain_size),
            domain_y=(0.0, domain_size),
            domain_t=(0.0, 10.0)  # Long time for steady state
        )
        self.name = "NavierStokes2DLidDriven"
        self.lid_velocity = 1.0

    def boundary_condition(self, x, y, t):
        """
        Return boundary values for lid-driven cavity.

        Args:
            x, y, t: Coordinates

        Returns:
            dict with u, v values at boundaries
        """
        # Top lid (y = y_max)
        at_top = (y >= self.domain_y[1] - 1e-6)
        # Other walls
        at_bottom = (y <= self.domain_y[0] + 1e-6)
        at_left = (x <= self.domain_x[0] + 1e-6)
        at_right = (x >= self.domain_x[1] - 1e-6)

        u_bc = torch.where(at_top, self.lid_velocity, 0.0)
        v_bc = torch.zeros_like(y)

        return {'u': u_bc, 'v': v_bc}


class NavierStokes2DChannel(NavierStokes2D):
    """
    2D Channel Flow (Poiseuille flow).

    Pressure-driven flow between parallel plates.
    """

    def __init__(
        self,
        nu: float = 0.01,
        dp_dx: float = -1.0,  # Pressure gradient
        height: float = 1.0
    ):
        super().__init__(
            nu=nu,
            domain_x=(0.0, 4.0),
            domain_y=(0.0, height),
            domain_t=(0.0, 5.0)
        )
        self.dp_dx = dp_dx
        self.height = height
        self.name = "NavierStokes2DChannel"

    def analytical_solution(self, y):
        """
        Analytical Poiseuille profile: u(y) = (dp/dx)/(2ν) * y * (H - y)
        """
        return (-self.dp_dx / (2 * self.nu)) * y * (self.height - y)
