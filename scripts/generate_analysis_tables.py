"""
Generate analysis_tables.md from SPIKE models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from spike.models import SPIKE, PIKE, PINN
from spike.evaluation.residuals import compute_residual
from spike.evaluation.koopman import compute_koopman_r2, check_stability
from spike.evaluation.lyapunov import compute_lyapunov_metrics
from spike.diffeq.pdes import (
    HeatEquation, AdvectionEquation, BurgersEquation,
    AllenCahnEquation, KdVEquation, ReactionDiffusionEquation,
    KuramotoSivashinskyEquation, SchrodingerEquation,
    CahnHilliardEquation, Wave2D, Burgers2D, NavierStokes2D,
    NavierStokes2DLidDriven
)
from spike.diffeq.odes import LorenzSystem, SEIRModel

CHECKPOINT_DIR = Path(__file__).parent.parent / 'results' / 'models'
OUTPUT_FILE = Path(__file__).parent.parent / 'results' / 'analysis' / 'analysis_tables.md'

VARIANTS = ['pinn', 'pike_euler', 'pike_rk4', 'pike_expm', 'spike_expm']
LABELS = ['PINN', 'PIKE-Euler', 'PIKE-RK4', 'PIKE-EXPM', 'SPIKE-EXPM']
KOOPMAN_VARIANTS = ['pike_euler', 'pike_rk4', 'pike_expm', 'spike_expm']
KOOPMAN_LABELS = ['PIKE-Euler', 'PIKE-RK4', 'PIKE-EXPM', 'SPIKE-EXPM']

# System configurations
PDE_1D_SYSTEMS = {
    'heat': HeatEquation,
    'advection': AdvectionEquation,
    'burgers': BurgersEquation,
    'allen_cahn': AllenCahnEquation,
    'kdv': KdVEquation,
    'reaction_diffusion': ReactionDiffusionEquation,
    'cahn_hilliard': CahnHilliardEquation,
    'kuramoto_sivashinsky': KuramotoSivashinskyEquation,
    'schrodinger': SchrodingerEquation,
}

PDE_2D_SYSTEMS = {
    'wave_2d_base': Wave2D,
    'burgers_2d_base': Burgers2D,
    'navier_stokes_2d': NavierStokes2D,
    'navier_stokes_2d_lid_driven': NavierStokes2DLidDriven,
}

ODE_SYSTEMS = {
    'lorenz': LorenzSystem,
    'seir': SEIRModel,
}

# Systems to include in OOD tables (exclude cahn_hilliard, schrodinger, kuramoto_sivashinsky)
OOD_SYSTEMS = ['heat', 'advection', 'burgers', 'allen_cahn', 'kdv', 'reaction_diffusion']


def load_model(system, variant):
    """Load checkpoint."""
    path = CHECKPOINT_DIR / system / variant / 'best_model.pt'
    if not path.exists():
        return None

    ckpt = torch.load(path, map_location='cpu')
    state = ckpt['model_state_dict']
    config = ckpt['config']
    model_type = config.get('model_type', 'PINN')

    if model_type == 'PINN':
        model = PINN(
            input_dim=config['input_dim'], output_dim=config['output_dim'],
            hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
            activation=config.get('activation', 'tanh')
        )
        model.load_state_dict(state)
    else:
        ModelClass = SPIKE if model_type == 'SPIKE' else PIKE
        model = ModelClass(
            input_dim=config['input_dim'], output_dim=config['output_dim'],
            hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
            embedding_dim=config['embedding_dim'],
            embedding_type=config.get('embedding_type', 'augmented'),
            poly_degree=config.get('poly_degree', 2),
            mlp_hidden=config.get('mlp_hidden', 64),
            activation=config.get('activation', 'tanh'),
            integrator=config.get('integrator', 'expm')
        )
        model.load_state_dict(state)

    model.eval()
    return model


def fmt(v, precision=2):
    """Format value for table."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:.{precision}e}'


def fmt_r2(v):
    """Format R² value."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:.4f}'


def generate_tables():
    """Generate all analysis tables."""
    output = []
    output.append("# SPIKE: Comprehensive Evaluation Results\n")

    # ========== 1. In-Domain Physics Residual (1D PDEs) ===========
    output.append("## 1. In-Domain Physics Residual MSE (1D PDEs)\n")
    output.append("Domain: x in [0,1], t in [0,1]\n")
    output.append("| PDE | " + " | ".join(LABELS) + " |")
    output.append("|-----|" + "|".join(["------" for _ in LABELS]) + "|")

    for name, PDE_Class in PDE_1D_SYSTEMS.items():
        diffeq = PDE_Class()
        row = [name]
        for variant in VARIANTS:
            model = load_model(name, variant)
            if model:
                mse = compute_residual(model, diffeq, domain={'x': (0, 1), 't': (0, 1)})
                row.append(fmt(mse))
            else:
                row.append('N/A')
        output.append("| " + " | ".join(row) + " |")
        print(f"  1D In-Domain: {name} done")

    # ========== 2. OOD-Space MSE ===========
    output.append("\n## 2. OOD-Space Physics Residual MSE\n")
    output.append("Spatial extrapolation, t in [0,1]\n")

    for x_label, x_range in [("x in [-5, 0]", (-5, 0)), ("x in [1, 3]", (1, 3))]:
        output.append(f"### {x_label}\n")
        output.append("| PDE | " + " | ".join(LABELS) + " |")
        output.append("|-----|" + "|".join(["------" for _ in LABELS]) + "|")

        for name in OOD_SYSTEMS:
            diffeq = PDE_1D_SYSTEMS[name]()
            row = [name]
            for variant in VARIANTS:
                model = load_model(name, variant)
                if model:
                    mse = compute_residual(model, diffeq, domain={'x': x_range, 't': (0, 1)})
                    row.append(fmt(mse))
                else:
                    row.append('N/A')
            output.append("| " + " | ".join(row) + " |")
        output.append("")
        print(f"  OOD-Space {x_label} done")

    # ========== 3. OOD-Time MSE ===========
    output.append("\n## 3. OOD-Time Physics Residual MSE\n")
    output.append("Temporal extrapolation, x in [0,1]\n")

    for t_label, t_range in [("t in [1, 3]", (1, 3)), ("t in [3, 5]", (3, 5))]:
        output.append(f"### {t_label}\n")
        output.append("| PDE | " + " | ".join(LABELS) + " |")
        output.append("|-----|" + "|".join(["------" for _ in LABELS]) + "|")

        for name in OOD_SYSTEMS:
            diffeq = PDE_1D_SYSTEMS[name]()
            row = [name]
            for variant in VARIANTS:
                model = load_model(name, variant)
                if model:
                    mse = compute_residual(model, diffeq, domain={'x': (0, 1), 't': t_range})
                    row.append(fmt(mse))
                else:
                    row.append('N/A')
            output.append("| " + " | ".join(row) + " |")
        output.append("")
        print(f"  OOD-Time {t_label} done")

    # ========== 4. Koopman Latent R² ===========
    output.append("\n## 4. Koopman Latent R² (PIKE/SPIKE only)\n")
    output.append("| System | " + " | ".join(KOOPMAN_LABELS) + " |")
    output.append("|--------|" + "|".join(["------" for _ in KOOPMAN_LABELS]) + "|")

    ALL_SYSTEMS = {**PDE_1D_SYSTEMS, **PDE_2D_SYSTEMS, **ODE_SYSTEMS}
    for name, DiffEq_Class in ALL_SYSTEMS.items():
        diffeq = DiffEq_Class()
        row = [name]
        for variant in KOOPMAN_VARIANTS:
            model = load_model(name, variant)
            if model and hasattr(model, 'koopman'):
                r2 = compute_koopman_r2(model, diffeq)
                row.append(fmt_r2(r2))
            else:
                row.append('N/A')
        output.append("| " + " | ".join(row) + " |")
    print("  Koopman R² done")

    # ========== 5. ODE Systems ===========
    output.append("\n## 5. ODE Systems\n")

    output.append("### In-Domain (t in [0,1])\n")
    output.append("| System | " + " | ".join(LABELS) + " |")
    output.append("|--------|" + "|".join(["------" for _ in LABELS]) + "|")

    for name, ODE_Class in ODE_SYSTEMS.items():
        diffeq = ODE_Class()
        row = [name]
        for variant in VARIANTS:
            model = load_model(name, variant)
            if model:
                mse = compute_residual(model, diffeq, domain={'t': (0, 1)})
                row.append(fmt(mse))
            else:
                row.append('N/A')
        output.append("| " + " | ".join(row) + " |")
    print("  ODE In-Domain done")

    output.append("\n### OOD-Time (t in [1,3])\n")
    output.append("| System | " + " | ".join(LABELS) + " |")
    output.append("|--------|" + "|".join(["------" for _ in LABELS]) + "|")

    for name, ODE_Class in ODE_SYSTEMS.items():
        diffeq = ODE_Class()
        row = [name]
        for variant in VARIANTS:
            model = load_model(name, variant)
            if model:
                mse = compute_residual(model, diffeq, domain={'t': (1, 3)})
                row.append(fmt(mse))
            else:
                row.append('N/A')
        output.append("| " + " | ".join(row) + " |")
    print("  ODE OOD-Time done")

    # ========== 6. 2D PDEs ===========
    output.append("\n## 6. 2D PDEs\n")

    output.append("### In-Domain (x,y,t in [0,1])\n")
    output.append("| PDE | " + " | ".join(LABELS) + " |")
    output.append("|-----|" + "|".join(["------" for _ in LABELS]) + "|")

    for name, PDE_Class in PDE_2D_SYSTEMS.items():
        diffeq = PDE_Class()
        row = [name]
        for variant in VARIANTS:
            model = load_model(name, variant)
            if model:
                mse = compute_residual(model, diffeq, domain={'x': (0, 1), 'y': (0, 1), 't': (0, 1)})
                row.append(fmt(mse))
            else:
                row.append('N/A')
        output.append("| " + " | ".join(row) + " |")
    print("  2D In-Domain done")

    # OOD-Space for open-domain 2D PDEs (Wave, Burgers: extend both x,y)
    output.append("\n### OOD-Space (x,y in [1,2], t in [0,1])\n")
    output.append("*For open-domain PDEs (Wave, Burgers). Not applicable to bounded-domain flows.*\n")
    output.append("| PDE | " + " | ".join(LABELS) + " |")
    output.append("|-----|" + "|".join(["------" for _ in LABELS]) + "|")

    for name in ['wave_2d_base', 'burgers_2d_base']:
        diffeq = PDE_2D_SYSTEMS[name]()
        row = [name]
        for variant in VARIANTS:
            model = load_model(name, variant)
            if model:
                mse = compute_residual(model, diffeq, domain={'x': (1, 2), 'y': (1, 2), 't': (0, 1)})
                row.append(fmt(mse))
            else:
                row.append('N/A')
        output.append("| " + " | ".join(row) + " |")
    print("  2D OOD-Space (open-domain) done")

    # OOD-Space Downstream for Channel Flow (physically meaningful: x in [1,2], y in [0,1])
    output.append("\n### OOD-Space Downstream (Channel Flow: x in [1,2], y in [0,1], t in [0,1])\n")
    output.append("*Physically meaningful OOD for channel flow: downstream prediction within channel boundaries.*\n")
    output.append("| PDE | " + " | ".join(LABELS) + " |")
    output.append("|-----|" + "|".join(["------" for _ in LABELS]) + "|")

    diffeq = PDE_2D_SYSTEMS['navier_stokes_2d']()
    row = ['navier_stokes_2d']
    for variant in VARIANTS:
        model = load_model('navier_stokes_2d', variant)
        if model:
            mse = compute_residual(model, diffeq, domain={'x': (1, 2), 'y': (0, 1), 't': (0, 1)})
            row.append(fmt(mse))
        else:
            row.append('N/A')
    output.append("| " + " | ".join(row) + " |")
    print("  2D OOD-Space Downstream (channel flow) done")

    # Note about lid-driven cavity
    output.append("\n*Note: OOD-Space is not physically meaningful for lid-driven cavity (bounded domain with fixed walls).*\n")

    output.append("\n### OOD-Time (x,y in [0,1], t in [1,3])\n")
    output.append("| PDE | " + " | ".join(LABELS) + " |")
    output.append("|-----|" + "|".join(["------" for _ in LABELS]) + "|")

    for name, PDE_Class in PDE_2D_SYSTEMS.items():
        diffeq = PDE_Class()
        row = [name]
        for variant in VARIANTS:
            model = load_model(name, variant)
            if model:
                mse = compute_residual(model, diffeq, domain={'x': (0, 1), 'y': (0, 1), 't': (1, 3)})
                row.append(fmt(mse))
            else:
                row.append('N/A')
        output.append("| " + " | ".join(row) + " |")
    print("  2D OOD-Time done")

    # ========== 7. Koopman Stability ===========
    output.append("\n## 7. Koopman Stability (max Re(lambda) <= 0.01)\n")
    output.append("| System | " + " | ".join(KOOPMAN_LABELS) + " |")
    output.append("|--------|" + "|".join(["------" for _ in KOOPMAN_LABELS]) + "|")

    for name in ALL_SYSTEMS.keys():
        row = [name]
        for variant in KOOPMAN_VARIANTS:
            model = load_model(name, variant)
            if model and hasattr(model, 'koopman'):
                stable, max_eig = check_stability(model)
                row.append("/" if stable else "x")
            else:
                row.append('N/A')
        output.append("| " + " | ".join(row) + " |")
    print("  Stability done")

    # ========== 8. Lyapunov Analysis (Chaotic Systems) ===========
    output.append("\n## 8. Lyapunov Analysis (Chaotic Systems)\n")
    output.append("Valid prediction time and tau ratio for Lorenz system (tau_lambda = 1.1s)\n")
    output.append("| Metric | " + " | ".join(LABELS) + " |")
    output.append("|--------|" + "|".join(["------" for _ in LABELS]) + "|")

    # Lorenz Lyapunov metrics
    lorenz = LorenzSystem()
    lyap_results = {}
    for variant in VARIANTS:
        model = load_model('lorenz', variant)
        if model:
            metrics = compute_lyapunov_metrics(
                model, lorenz,
                t_train=(0, 25),
                t_ood=(25, 35),
                tau_lambda=1.1,
                n_points=1000,
                error_threshold=0.5
            )
            lyap_results[variant] = metrics
        else:
            lyap_results[variant] = {}

    # Valid Time row
    row = ['Valid Time (s)']
    for variant in VARIANTS:
        vt = lyap_results[variant].get('valid_time', float('nan'))
        row.append(f'{vt:.2f}' if not np.isnan(vt) else 'N/A')
    output.append("| " + " | ".join(row) + " |")

    # Tau Ratio row
    row = ['Tau Ratio']
    for variant in VARIANTS:
        tr = lyap_results[variant].get('tau_ratio', float('nan'))
        row.append(f'{tr:.2f}' if not np.isnan(tr) else 'N/A')
    output.append("| " + " | ".join(row) + " |")

    # Short-term MSE row
    row = ['Short-term MSE']
    for variant in VARIANTS:
        st = lyap_results[variant].get('short_term_mse', float('nan'))
        row.append(fmt(st) if not np.isnan(st) else 'N/A')
    output.append("| " + " | ".join(row) + " |")

    print("  Lyapunov done")

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    print("Generating analysis tables using SPIKE modules...")
    generate_tables()
