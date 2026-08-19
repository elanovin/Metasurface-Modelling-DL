"""Physics verification for the transfer-matrix S-parameter solver.

Each test checks a property that follows from Maxwell's equations and can be
verified independently of the implementation, so a passing suite is evidence
that the solver is physically correct rather than merely self-consistent.

Run with:  python -m pytest tests/ -v      (or simply: python tests/test_tmm_solver.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tmm_solver import (  # noqa: E402
    ETA0,
    C0,
    single_slab_reference,
    stack_s_parameters,
)

RNG = np.random.default_rng(20260819)


def _random_lossless_stack(n=400, n_layers=3):
    f = RNG.uniform(1e9, 40e9, n)
    theta = RNG.uniform(0.0, np.deg2rad(85.0), n)
    eps_r = RNG.uniform(1.5, 12.0, (n, n_layers))
    d = RNG.uniform(0.1e-3, 5.0e-3, (n, n_layers))
    return f, theta, eps_r, d


# --------------------------------------------------------------------------
# 1. A vacuum layer must be invisible apart from the propagation phase.
# --------------------------------------------------------------------------
def test_vacuum_layer_is_transparent():
    f = np.array([10e9, 25e9])
    theta = np.deg2rad([0.0, 55.0])
    eps_r = np.ones((2, 1))
    d = np.full((2, 1), 3e-3)

    for te in (True, False):
        s11, s21 = stack_s_parameters(f, theta, eps_r, d, te=te)
        kz0 = (2 * np.pi * f / C0) * np.cos(theta)
        assert np.allclose(s11, 0.0, atol=1e-12)
        assert np.allclose(s21, np.exp(-1j * kz0 * d[:, 0]), atol=1e-12)


# --------------------------------------------------------------------------
# 2. A half-wave slab is reflectionless at its design frequency.
# --------------------------------------------------------------------------
def test_half_wave_slab_is_reflectionless():
    f = 10e9
    eps_r = 4.0
    theta = 0.0
    kz = (2 * np.pi * f / C0) * np.sqrt(eps_r)
    d = np.pi / kz                                     # kz * d = pi

    s11, s21 = stack_s_parameters([f], [theta], [[eps_r]], [[d]])
    assert abs(s11[0]) < 1e-12
    assert abs(abs(s21[0]) - 1.0) < 1e-12


# --------------------------------------------------------------------------
# 3. A quarter-wave slab has a reflection coefficient known in closed form.
# --------------------------------------------------------------------------
def test_quarter_wave_slab_closed_form():
    f, eps_r, theta = 10e9, 4.0, 0.0
    kz = (2 * np.pi * f / C0) * np.sqrt(eps_r)
    d = (np.pi / 2) / kz                               # kz * d = pi/2

    s11, s21 = stack_s_parameters([f], [theta], [[eps_r]], [[d]])
    expected_s11 = (1.0 - eps_r) / (1.0 + eps_r)       # = -0.6 for eps_r = 4
    assert abs(s11[0] - expected_s11) < 1e-12
    assert abs(abs(s21[0]) - 0.8) < 1e-12


# --------------------------------------------------------------------------
# 4. Energy conservation: a lossless stack must satisfy |S11|^2 + |S21|^2 = 1.
# --------------------------------------------------------------------------
def test_energy_is_conserved_in_lossless_stacks():
    f, theta, eps_r, d = _random_lossless_stack()
    for te in (True, False):
        s11, s21 = stack_s_parameters(f, theta, eps_r, d, te=te)
        total = np.abs(s11) ** 2 + np.abs(s21) ** 2
        assert np.max(np.abs(total - 1.0)) < 1e-10


# --------------------------------------------------------------------------
# 5. Passivity: a lossy stack must absorb, never generate, power.
# --------------------------------------------------------------------------
def test_lossy_stack_is_passive_and_absorbs():
    f, theta, eps_r, d = _random_lossless_stack()
    sigma = RNG.uniform(0.05, 5.0, eps_r.shape)
    for te in (True, False):
        s11, s21 = stack_s_parameters(f, theta, eps_r, d, sigma=sigma, te=te)
        total = np.abs(s11) ** 2 + np.abs(s21) ** 2
        assert np.all(total < 1.0 + 1e-12)             # passive
        assert np.median(total) < 0.999                # genuinely lossy


# --------------------------------------------------------------------------
# 6. Agreement with an independently derived closed-form single-slab result.
# --------------------------------------------------------------------------
def test_matches_independent_single_slab_formula():
    n = 500
    f = RNG.uniform(1e9, 40e9, n)
    theta = RNG.uniform(0.0, np.deg2rad(85.0), n)
    eps_r = RNG.uniform(1.5, 12.0, n)
    d = RNG.uniform(0.1e-3, 5.0e-3, n)
    sigma = RNG.choice([0.0, 0.3, 2.0], n)

    for te in (True, False):
        s11, s21 = stack_s_parameters(
            f, theta, eps_r[:, None], d[:, None], sigma=sigma[:, None], te=te)
        g_ref, t_ref = single_slab_reference(
            f, theta, eps_r, d, sigma=sigma, te=te)
        assert np.max(np.abs(s11 - g_ref)) < 1e-10
        assert np.max(np.abs(s21 - t_ref)) < 1e-10


# --------------------------------------------------------------------------
# 7. At normal incidence the two polarizations must coincide.
# --------------------------------------------------------------------------
def test_te_and_tm_agree_at_normal_incidence():
    _, _, eps_r, d = _random_lossless_stack(n=200)
    f = RNG.uniform(1e9, 40e9, 200)
    theta = np.zeros(200)
    sigma = RNG.uniform(0.0, 2.0, eps_r.shape)

    s_te = stack_s_parameters(f, theta, eps_r, d, sigma=sigma, te=True)
    s_tm = stack_s_parameters(f, theta, eps_r, d, sigma=sigma, te=False)
    assert np.max(np.abs(s_te[0] - s_tm[0])) < 1e-10
    assert np.max(np.abs(s_te[1] - s_tm[1])) < 1e-10


# --------------------------------------------------------------------------
# 8. Layer order matters for reflection but not for transmission (reciprocity
#    of a reciprocal cascade: S21 is invariant under reversal of the stack).
# --------------------------------------------------------------------------
def test_transmission_is_invariant_under_stack_reversal():
    f, theta, eps_r, d = _random_lossless_stack()
    sigma = RNG.uniform(0.0, 1.0, eps_r.shape)

    _, s21 = stack_s_parameters(f, theta, eps_r, d, sigma=sigma)
    _, s21_rev = stack_s_parameters(
        f, theta, eps_r[:, ::-1], d[:, ::-1], sigma=sigma[:, ::-1])
    assert np.max(np.abs(s21 - s21_rev)) < 1e-10


# --------------------------------------------------------------------------
# 9. Grazing incidence limit: reflection tends to total.
# --------------------------------------------------------------------------
def test_grazing_incidence_tends_to_total_reflection():
    f = np.full(5, 10e9)
    theta = np.deg2rad([89.0, 89.5, 89.9, 89.99, 89.999])
    eps_r = np.full((5, 1), 4.0)
    d = np.full((5, 1), 2e-3)

    s11, _ = stack_s_parameters(f, theta, eps_r, d)
    assert np.all(np.diff(np.abs(s11)) > -1e-12)       # monotonically increasing
    assert abs(s11[-1]) > 0.999


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print()
    print("all physics checks passed" if failures == 0 else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
