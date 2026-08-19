"""Feature construction, including an optional physics-informed feature set.

Two feature sets are provided so that their effect can be compared directly:

`raw`
    The 12 physical parameters as sampled: permittivity, conductivity and
    thickness of each layer, plus frequency, incidence angle and polarization.

`physics`
    The same 12 parameters plus, for each layer, three derived quantities
    built from the normal wavenumber kz:

        cos(beta_m d_m),  sin(beta_m d_m),  exp(-alpha_m d_m)

    where beta = Re(kz) and alpha = -Im(kz), so that beta*d is the electrical
    thickness of the layer and alpha*d is its attenuation over one pass.

The motivation is physical rather than empirical. The S-parameters of a stack
depend on the electrical thickness of each layer, not on its geometric
thickness and the frequency separately, and they oscillate as that electrical
thickness passes through multiples of pi. A network given only d and f has to
learn this product and its oscillation from scratch; supplying it directly
removes the hardest part of the mapping.

Comparing the two sets is a small ablation: it isolates how much of the
regression difficulty comes from the oscillatory phase dependence alone.
"""

import numpy as np

from tmm_solver import wave_numbers

N_LAYERS = 3


def _physics_block(X):
    """Derived per-layer phase and attenuation features from a raw feature array."""
    eps_r = X[:, 0:3]
    sigma = X[:, 3:6]
    d = X[:, 6:9] * 1e-3            # mm -> m
    f = X[:, 9] * 1e9               # GHz -> Hz
    theta = np.deg2rad(X[:, 10])

    blocks = []
    for m in range(N_LAYERS):
        kz, _, _, _ = wave_numbers(
            f, theta, eps_r[:, m], np.ones_like(eps_r[:, m]), sigma[:, m])
        beta = np.real(kz)
        alpha = -np.imag(kz)                       # >= 0 on the passive branch
        phase = beta * d[:, m]
        atten = np.exp(-np.clip(alpha * d[:, m], 0.0, 50.0))
        blocks.append(np.column_stack([np.cos(phase), np.sin(phase), atten]))
    return np.hstack(blocks)


def build(X, kind="physics"):
    """Return the feature matrix for the requested feature set."""
    if kind == "raw":
        return X
    if kind == "physics":
        return np.hstack([X, _physics_block(X)])
    raise ValueError(f"unknown feature set: {kind!r} (expected 'raw' or 'physics')")


def names(base_names, kind="physics"):
    base_names = list(base_names)
    if kind == "raw":
        return base_names
    derived = []
    for m in range(1, N_LAYERS + 1):
        derived += [f"cos_beta{m}d{m}", f"sin_beta{m}d{m}", f"atten{m}"]
    return base_names + derived
