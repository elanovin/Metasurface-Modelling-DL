# S-Parameter Surrogate Modelling for Multilayer Structures

A neural-network surrogate that predicts the complex reflection and transmission
coefficients (S11, S21) of a planar multilayer stack, trained on ground truth
from an exact transfer-matrix electromagnetic solver.

> **Note on scope and history.** Earlier versions of this repository used
> placeholder analytic expressions in place of an electromagnetic solver, and
> the results were therefore not physically meaningful. That generator has been
> removed and replaced with the exact transfer-matrix solver documented below,
> together with a physics verification suite. The geometry modelled here is a
> planar multilayer stack, not a patterned metasurface unit cell; extending it
> to patterned cells requires full-wave simulation with periodic boundary
> conditions and is listed under future work.

---

## The physics

The structure is a stack of `N` homogeneous, isotropic layers embedded in free
space, infinite in the transverse plane, illuminated by a plane wave at
incidence angle `theta0`. Each layer has a relative permittivity, a relative
permeability, an electrical conductivity and a physical thickness.

Each layer is represented by the ABCD matrix of an equivalent transmission-line
section, in which the line voltage and current correspond to the tangential
electric and magnetic field components:

```
        | cos(kz_m d_m)          j Z_m sin(kz_m d_m) |
M_m  =  |                                            |
        | j sin(kz_m d_m) / Z_m  cos(kz_m d_m)       |
```

with the normal wavenumber `kz_m = sqrt(k_m^2 - kx^2)`, the transverse
wavenumber `kx = k0 sin(theta0)` conserved across every interface, and the
transverse wave impedance

```
    TE (perpendicular):  Z_m = omega mu_m / kz_m
    TM (parallel):       Z_m = kz_m / (omega eps_m)
```

Loss enters through the complex permittivity `eps = eps0 eps_r - j sigma/omega`.
The stack matrix is the ordered product `M = M_1 M_2 ... M_N`, and the
S-parameters follow from the standard ABCD-to-S conversion using the free-space
wave impedance at the incidence angle as the reference impedance on both ports.

Tangential-field continuity is imposed exactly at every interface and no
expansion is truncated, so this is the exact solution of Maxwell's equations for
this geometry — multiple internal reflections and the resulting resonances are
captured exactly, not approximated.

## Verification

The solver is checked against properties that follow from Maxwell's equations
and can be verified independently of the implementation:

| Check | Property |
| --- | --- |
| Vacuum layer | `S11 = 0`, `S21 = exp(-j kz d)` |
| Half-wave slab | reflectionless at the design frequency |
| Quarter-wave slab | `S11 = (1 - eps_r)/(1 + eps_r)` in closed form |
| Lossless stacks | `abs(S11)^2 + abs(S21)^2 = 1` (energy conservation) |
| Lossy stacks | `abs(S11)^2 + abs(S21)^2 < 1` (passivity) |
| Single slab | agreement with the independently derived Airy multiple-reflection formula |
| Normal incidence | TE and TM coincide |
| Stack reversal | `S21` invariant, as required for a reciprocal cascade |
| Grazing incidence | reflection tends monotonically to unity |

```bash
python -m pytest tests/ -v        # or: python tests/test_tmm_solver.py
```

All checks pass to within `1e-10`.

## Dataset

300,000 random three-layer configurations, generated in about two seconds.

| Parameter | Range | Sampling |
| --- | --- | --- |
| Relative permittivity, per layer | 1.5 – 12.0 | uniform |
| Conductivity, per layer | 1e-3 – 5.0 S/m | log-uniform, 25% of stacks exactly lossless |
| Thickness, per layer | 0.1 – 5.0 mm | uniform |
| Frequency | 1 – 40 GHz | uniform |
| Incidence angle | 0° – 85° | uniform |
| Polarization | TE / TM | uniform |

Conductivity is sampled logarithmically because loss spans several orders of
magnitude, and a lossless subset is retained so that the network also sees the
energy-conserving, resonance-dominated regime, which is the harder one to learn.

## Feature sets and ablation

Two feature sets are compared. The physical motivation is that the
S-parameters depend on the *electrical* thickness of each layer, not on its
geometric thickness and the frequency separately, and that they oscillate as
that electrical thickness passes through multiples of pi. A network given only
`d` and `f` has to learn that product and its oscillation from scratch.

- **`raw`** — the 12 sampled physical parameters.
- **`physics`** — the same 12, plus `cos(beta_m d_m)`, `sin(beta_m d_m)` and
  `exp(-alpha_m d_m)` for each layer, where `beta = Re(kz)` and `alpha = -Im(kz)`.

| Feature set | Features | Test MAE | Test RMSE | Test R² | Passivity violations |
| --- | --- | --- | --- | --- | --- |
| `raw` | 12 | 0.0539 | 0.0834 | 0.9552 | 15.4% |
| `physics` | 21 | **0.0240** | **0.0337** | **0.9927** | 9.4% |

Supplying the electrical thickness explicitly reduces the error by a factor of
about 2.2, which isolates how much of the regression difficulty comes from the
oscillatory phase dependence alone.

## Model

Fully connected regressor, hidden widths `[512, 256, 128, 64]`, ReLU with
He-normal initialisation, batch normalisation, and dropout decreasing across
depth (0.4 / 0.3 / 0.2 / 0). Four linear outputs: `Re(S11)`, `Im(S11)`,
`Re(S21)`, `Im(S21)`. Adam at `1e-3`, MSE loss, min–max input scaling,
80/10/10 train/validation/test split, early stopping and learning-rate
reduction on plateau.

Keras is used when TensorFlow is available; the script otherwise falls back to
a scikit-learn MLP so the repository runs without a deep-learning install. The
numbers reported above are from the scikit-learn path.

## Honest limitations

- **The surrogate is not physically constrained.** Nothing forces the network
  to respect `abs(S11)^2 + abs(S21)^2 <= 1`, and 9.4% of test predictions
  violate it, with a maximum predicted energy of 1.24. This is reported rather
  than hidden; enforcing passivity through the output parameterisation is the
  clearest next improvement.
- **Planar layers only.** This is not a patterned metasurface unit cell. A
  patterned cell needs full-wave simulation with periodic boundary conditions
  and Floquet ports.
- **Fixed three layers.** The solver handles any `N`; the trained model does not
  generalise beyond the layer count it was trained on.
- **No timing benchmark yet.** The surrogate is a batched forward pass and the
  solver is a per-configuration matrix product, but the break-even point —
  including the cost of generating training data — has not been measured, and
  no speed-up figure is claimed here without it.

## Usage

```bash
pip install -r requirements.txt

python -m pytest tests/ -v                                   # verify the physics
python src/data_generator.py --samples 300000                # build the dataset
python src/train.py --features physics                       # train
python src/train.py --features raw                           # ablation

python src/inference.py --eps 4.4 2.2 9.8 --sigma 0 0.1 0 \
    --thickness 1.5 0.8 2.0 --freq 10 --theta 30 --pol TE --compare
```

`--compare` evaluates the exact solver alongside the surrogate and prints the
error, so any prediction can be checked directly against ground truth.

## Repository layout

```
src/tmm_solver.py       exact transfer-matrix S-parameter solver
src/features.py         raw and physics-informed feature construction
src/data_generator.py   parameter sampling and dataset generation
src/train.py            training and evaluation
src/inference.py        single-configuration prediction, optional solver comparison
tests/                  physics verification suite
```

## Future work

- Enforce passivity in the output parameterisation.
- Replace the planar stack with full-wave unit-cell simulations (periodic
  boundary conditions, Floquet ports) to model patterned metasurfaces.
- Variable layer count.
- Benchmark solver and surrogate on identical hardware and report the
  break-even point including data-generation cost.
- Inverse design: target response to geometry.

## Related work

Physically validated surrogate modelling of surface impedance and effective
surface resistivity for multilayer structures, using the exact multilayer
field-matching formulation with the Leontovich impedance boundary condition as
ground truth:
[deep-learning-surface-impedance](https://github.com/elanovin/deep-learning-surface-impedance)
(published at IEEE SMAP 2026).

## License

MIT
