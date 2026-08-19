"""Dataset generation for the S-parameter surrogate.

Physical parameters are sampled over documented ranges and the corresponding
S11 and S21 are computed with the exact transfer-matrix solver in
`tmm_solver.py`. The ground truth is therefore an exact solution of Maxwell's
equations for this geometry, not an empirical fit or a placeholder function.

Usage
-----
    python src/data_generator.py --samples 300000 --out data/dataset.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmm_solver import stack_s_parameters  # noqa: E402

N_LAYERS = 3

# Sampling ranges. Chosen to cover electrically thin and electrically thick
# regimes, low- and high-contrast dielectrics, and lossless through lossy
# behaviour, so that the dataset contains both smooth responses and the sharp
# resonances produced by multiple internal reflections.
RANGES = {
    "eps_r": (1.5, 12.0),        # relative permittivity, dimensionless
    "sigma": (1.0e-3, 5.0),      # electrical conductivity, S/m (log-sampled)
    "d": (0.1e-3, 5.0e-3),       # layer thickness, m
    "f": (1.0e9, 40.0e9),        # frequency, Hz
    "theta": (0.0, np.deg2rad(85.0)),   # incidence angle, rad
}

# Fraction of stacks made exactly lossless. Without this the dataset would be
# dominated by strongly absorbing configurations and the network would rarely
# see the energy-conserving, resonance-dominated regime, which is the harder
# and more physically interesting one to learn.
LOSSLESS_FRACTION = 0.25

FEATURE_NAMES = (
    [f"eps_r{i + 1}" for i in range(N_LAYERS)]
    + [f"sigma{i + 1}" for i in range(N_LAYERS)]
    + [f"d{i + 1}_mm" for i in range(N_LAYERS)]
    + ["freq_GHz", "theta_deg", "is_TE"]
)

TARGET_NAMES = ["Re_S11", "Im_S11", "Re_S21", "Im_S21"]


def generate(n_samples, seed=0):
    """Return (X, y, s11, s21) for `n_samples` random multilayer configurations."""
    rng = np.random.default_rng(seed)

    eps_r = rng.uniform(*RANGES["eps_r"], (n_samples, N_LAYERS))
    d = rng.uniform(*RANGES["d"], (n_samples, N_LAYERS))

    # Conductivity spans several orders of magnitude, so it is sampled
    # logarithmically; a subset of stacks is made exactly lossless.
    lo, hi = np.log10(RANGES["sigma"])
    sigma = 10.0 ** rng.uniform(lo, hi, (n_samples, N_LAYERS))
    sigma[rng.random(n_samples) < LOSSLESS_FRACTION, :] = 0.0
    f = rng.uniform(*RANGES["f"], n_samples)
    theta = rng.uniform(*RANGES["theta"], n_samples)
    te = rng.integers(0, 2, n_samples).astype(bool)

    s11, s21 = stack_s_parameters(f, theta, eps_r, d, sigma=sigma, te=te)

    # Features are stored in human-readable units (mm, GHz, degrees) so the
    # dataset is inspectable; scaling happens at training time.
    X = np.column_stack([
        eps_r, sigma, d * 1e3, f * 1e-9, np.rad2deg(theta), te.astype(float)
    ])
    y = np.column_stack([s11.real, s11.imag, s21.real, s21.imag])
    return X, y, s11, s21


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/dataset.npz")
    args = p.parse_args()

    X, y, s11, s21 = generate(args.samples, args.seed)

    # Sanity check on the generated physics before anything is written to disk.
    energy = np.abs(s11) ** 2 + np.abs(s21) ** 2
    assert np.all(energy < 1.0 + 1e-9), "generated data violates passivity"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, X=X, y=y,
        feature_names=np.array(FEATURE_NAMES),
        target_names=np.array(TARGET_NAMES),
    )

    print(f"wrote {args.out}")
    print(f"  samples          : {len(X):,}")
    print(f"  features         : {X.shape[1]}  {FEATURE_NAMES}")
    print(f"  targets          : {y.shape[1]}  {TARGET_NAMES}")
    print(f"  |S11| range      : {np.abs(s11).min():.4f} .. {np.abs(s11).max():.4f}")
    print(f"  |S21| range      : {np.abs(s21).min():.4f} .. {np.abs(s21).max():.4f}")
    print(f"  absorbed power   : median {1 - np.median(energy):.4f}, "
          f"max {1 - energy.min():.4f}")


if __name__ == "__main__":
    main()
