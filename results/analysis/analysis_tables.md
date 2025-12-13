# SPIKE: Comprehensive Evaluation Results

## 1. In-Domain Physics Residual MSE (1D PDEs)

Domain: x in [0,1], t in [0,1]

| PDE | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|-----|------|------|------|------|------|
| heat | 1.01e-05 | 6.95e-06 | 1.20e-05 | 6.02e-06 | 8.67e-06 |
| advection | 5.04e-07 | 8.89e-05 | 4.45e-07 | 4.88e-07 | 3.90e-07 |
| burgers | 9.68e-05 | 4.63e-05 | 3.44e-05 | 3.23e-05 | 3.23e-05 |
| allen_cahn | 1.19e-05 | 1.56e-05 | 1.44e-05 | 2.34e-05 | 7.08e-05 |
| kdv | 5.23e-02 | 4.75e-02 | 5.24e-02 | 5.06e-02 | 5.14e-02 |
| reaction_diffusion | 5.02e-05 | 2.67e-04 | 4.80e-05 | 4.63e-05 | 1.51e-04 |
| cahn_hilliard | 3.53e-01 | 2.61e-07 | 3.50e-01 | 3.46e-01 | 1.22e-01 |
| kuramoto_sivashinsky | 1.83e+01 | 8.77e+01 | 6.75e+02 | 2.93e+02 | 1.91e+01 |
| schrodinger | 2.50e+01 | 2.53e+01 | 2.49e+01 | 2.47e+01 | 1.20e+01 |

## 2. OOD-Space Physics Residual MSE

Spatial extrapolation, t in [0,1]

### x in [-5, 0]

| PDE | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|-----|------|------|------|------|------|
| heat | 6.49e-04 | 7.67e-04 | 6.71e-04 | 6.60e-04 | 6.49e-04 |
| advection | 2.64e-07 | 4.28e-06 | 2.63e-07 | 2.50e-07 | 3.02e-07 |
| burgers | 9.63e-03 | 1.30e-02 | 1.31e-02 | 1.53e-02 | 1.53e-02 |
| allen_cahn | 2.03e-02 | 2.95e-02 | 2.86e-02 | 4.86e-02 | 5.10e-02 |
| kdv | 2.76e-02 | 2.49e-02 | 2.76e-02 | 2.55e-02 | 2.61e-02 |
| reaction_diffusion | 3.71e-05 | 4.60e-04 | 3.06e-05 | 7.70e-05 | 2.77e-05 |

### x in [1, 3]

| PDE | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|-----|------|------|------|------|------|
| heat | 1.64e-03 | 1.09e-03 | 1.71e-03 | 1.69e-03 | 1.66e-03 |
| advection | 4.50e-07 | 1.86e-05 | 2.87e-07 | 3.13e-07 | 1.98e-07 |
| burgers | 5.90e-03 | 1.80e-02 | 6.59e-03 | 6.10e-03 | 6.10e-03 |
| allen_cahn | 4.50e-02 | 5.89e-02 | 5.02e-02 | 5.85e-02 | 6.17e-02 |
| kdv | 2.63e-02 | 2.57e-02 | 2.64e-02 | 2.64e-02 | 2.63e-02 |
| reaction_diffusion | 5.55e-05 | 2.71e-03 | 5.82e-05 | 7.29e-05 | 3.97e-05 |


## 3. OOD-Time Physics Residual MSE

Temporal extrapolation, x in [0,1]

### t in [1, 3]

| PDE | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|-----|------|------|------|------|------|
| heat | 1.82e-04 | 1.64e-04 | 1.94e-04 | 1.80e-04 | 1.79e-04 |
| advection | 3.10e-07 | 1.23e-05 | 3.72e-07 | 3.20e-07 | 3.89e-07 |
| burgers | 8.10e-02 | 3.09e-02 | 1.12e-01 | 8.93e-02 | 8.93e-02 |
| allen_cahn | 2.04e-02 | 2.33e-02 | 1.93e-02 | 2.37e-02 | 2.35e-02 |
| kdv | 2.28e-02 | 3.48e-02 | 2.17e-02 | 1.78e-02 | 1.89e-02 |
| reaction_diffusion | 2.00e-03 | 4.20e-03 | 2.10e-03 | 2.27e-03 | 2.31e-03 |

### t in [3, 5]

| PDE | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|-----|------|------|------|------|------|
| heat | 7.52e-04 | 2.17e-03 | 7.00e-04 | 7.97e-04 | 8.03e-04 |
| advection | 4.05e-10 | 2.96e-09 | 6.02e-10 | 4.48e-10 | 1.06e-09 |
| burgers | 2.77e-02 | 1.15e-02 | 2.97e-02 | 2.91e-02 | 2.91e-02 |
| allen_cahn | 6.21e-02 | 1.11e-01 | 6.46e-02 | 9.76e-02 | 9.58e-02 |
| kdv | 8.89e-03 | 2.49e-02 | 9.99e-03 | 1.42e-03 | 1.81e-03 |
| reaction_diffusion | 9.72e-03 | 1.73e-02 | 1.06e-02 | 1.17e-02 | 1.12e-02 |


## 4. Koopman Latent R² (PIKE/SPIKE only)

| System | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|--------|------|------|------|------|
| heat | 0.9989 | 0.9989 | 0.9990 | 0.9990 |
| advection | 0.8895 | 0.9004 | 0.9003 | 0.9003 |
| burgers | 0.9699 | 0.9670 | 0.9695 | 0.9695 |
| allen_cahn | 0.9983 | 0.9983 | 0.9983 | 0.9982 |
| kdv | 0.9247 | 0.9481 | 0.9495 | 0.9471 |
| reaction_diffusion | 0.9827 | 0.9895 | 0.9895 | 0.9857 |
| cahn_hilliard | 0.9998 | -0.2336 | -0.1506 | 0.8419 |
| kuramoto_sivashinsky | 0.5261 | 0.7087 | 0.6314 | 0.7566 |
| schrodinger | 0.9188 | 0.7376 | 0.6257 | 0.7242 |
| wave_2d_base | 1.0000 | 0.1409 | -0.0979 | 0.6045 |
| burgers_2d_base | 0.9580 | 0.9503 | 0.9756 | 0.9550 |
| navier_stokes_2d | 0.8008 | 0.9041 | 0.8931 | 0.8750 |
| lorenz | -0.8540 | -0.4453 | -0.6355 | -0.6355 |
| seir | -1712199.6170 | 0.4553 | 0.5885 | 0.5885 |

## 5. ODE Systems

### In-Domain (t in [0,1])

| System | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|--------|------|------|------|------|------|
| lorenz | 5.48e+04 | 4.79e+04 | 5.03e+04 | 4.35e+04 | 4.35e+04 |
| seir | 4.70e-02 | 4.26e-02 | 5.26e-02 | 5.73e-02 | 5.73e-02 |

### OOD-Time (t in [1,3])

| System | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|--------|------|------|------|------|------|
| lorenz | 2.96e+01 | 1.19e+01 | 1.95e+01 | 1.29e+01 | 1.29e+01 |
| seir | 1.84e-03 | 1.42e-03 | 1.67e-03 | 2.10e-03 | 2.10e-03 |

## 6. 2D PDEs

### In-Domain (x,y,t in [0,1])

| PDE | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|-----|------|------|------|------|------|
| wave_2d_base | 1.72e+04 | 2.06e-04 | 2.17e+05 | 7.69e+04 | 5.11e-02 |
| burgers_2d_base | 9.66e-03 | 7.12e-03 | 4.92e-02 | 3.23e-02 | 9.34e-03 |
| navier_stokes_2d | 1.81e-01 | 1.43e-01 | 2.58e-01 | 3.59e-01 | 1.56e-01 |

### OOD-Space (x,y in [1,2], t in [0,1])

| PDE | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|-----|------|------|------|------|------|
| wave_2d_base | 1.14e+00 | 1.32e-04 | 1.35e+00 | 1.38e-01 | 5.20e-03 |
| burgers_2d_base | 6.19e-03 | 1.62e-04 | 2.42e-04 | 2.79e-04 | 2.99e-03 |
| navier_stokes_2d | 2.85e+00 | 4.50e-01 | 5.41e-01 | 1.39e+00 | 5.99e-01 |

### OOD-Time (x,y in [0,1], t in [1,3])

| PDE | PINN | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|-----|------|------|------|------|------|
| wave_2d_base | 3.19e+00 | 2.06e-04 | 6.23e-04 | 1.81e-02 | 1.16e-02 |
| burgers_2d_base | 7.87e-02 | 9.29e-02 | 6.80e-02 | 7.88e-02 | 6.12e-02 |
| navier_stokes_2d | 2.51e-03 | 2.30e-03 | 2.88e-03 | 3.21e-03 | 3.63e-03 |

## 7. Koopman Stability (max Re(lambda) <= 0.01)

| System | PIKE-Euler | PIKE-RK4 | PIKE-EXPM | SPIKE-EXPM |
|--------|------|------|------|------|
| heat | / | / | / | / |
| advection | x | / | / | / |
| burgers | x | / | / | / |
| allen_cahn | / | / | / | / |
| kdv | / | / | / | / |
| reaction_diffusion | x | / | / | / |
| cahn_hilliard | / | / | / | / |
| kuramoto_sivashinsky | / | / | / | / |
| schrodinger | / | / | / | / |
| wave_2d_base | / | / | / | / |
| burgers_2d_base | / | / | / | / |
| navier_stokes_2d | x | / | / | / |
| lorenz | x | / | / | / |
| seir | x | / | / | / |