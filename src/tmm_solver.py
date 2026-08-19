"""
Physically exact S-parameter solver for planar multilayer structures.

The stack is a sequence of N homogeneous, isotropic layers embedded in free
space, infinite in the transverse plane, illuminated by a plane wave at an
incidence angle theta0. Each layer is described by its relative permittivity,
relative permeability, electrical conductivity and physical thickness.

Method
------
Each layer is represented by the ABCD (transmission) matrix of an equivalent
transmission-line section, where the line voltage and current correspond to the
tangential electric and magnetic field components:

    M_m = [[ cos(kz_m d_m)          , j Z_m sin(kz_m d_m) ],
           [ j sin(kz_m d_m) / Z_m  , cos(kz_m d_m)       ]]

The stack matrix is the ordered product M = M_1 M_2 ... M_N, and the
S-parameters follow from the standard ABCD -> S conversion with the free-space
wave impedance at the incidence angle as the reference impedance on both ports.

Because the tangential-field continuity conditions are imposed exactly at every
interface and no expansion is truncated, the result is the exact solution of
Maxwell's equations for this geometry, not an approximation. Multiple internal
reflections and the resulting resonances are therefore captured exactly.

Conventions
-----------
Time dependence exp(+j omega t); waves propagating in +z vary as exp(-j kz z).
The branch of kz is chosen so that Im(kz) <= 0, i.e. a wave decays as it
propagates into a lossy medium.
"""

import numpy as np

EPS0 = 8.8541878128e-12          # F/m
MU0 = 4.0e-7 * np.pi             # H/m
C0 = 1.0 / np.sqrt(EPS0 * MU0)   # m/s
ETA0 = np.sqrt(MU0 / EPS0)       # Ohm, ~376.73


def _as_2d(a, n_samples, n_layers, name):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = np.broadcast_to(a, (n_samples, n_layers))
    if a.shape != (n_samples, n_layers):
        raise ValueError(
            f"{name} must have shape ({n_samples}, {n_layers}), got {a.shape}"
        )
    return a


def wave_numbers(f, theta0, eps_r, mu_r, sigma):
    """Normal wavenumber kz and the complex material constants of a medium.

    Parameters are broadcast against each other. Returns (kz, eps, mu, omega).
    """
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    k0 = omega / C0
    kx = k0 * np.sin(theta0)                      # conserved across interfaces

    eps = EPS0 * eps_r - 1j * sigma / omega       # complex permittivity
    mu = MU0 * mu_r

    k = omega * np.sqrt(mu * eps + 0j)            # bulk wavenumber
    kz = np.sqrt(k ** 2 - kx ** 2 + 0j)           # normal component
    kz = np.where(np.imag(kz) > 0.0, -kz, kz)     # passive branch: Im(kz) <= 0
    return kz, eps, mu, omega


def wave_impedance(kz, eps, mu, omega, te):
    """Transverse wave impedance. `te` is a boolean array: True for TE, False TM."""
    z_te = omega * mu / kz
    z_tm = kz / (omega * eps)
    return np.where(te, z_te, z_tm)


def stack_s_parameters(f, theta0, eps_r, d, sigma=None, mu_r=None, te=True):
    """S11 and S21 of a multilayer stack in free space.

    Parameters
    ----------
    f       : (M,)    frequency in Hz
    theta0  : (M,)    incidence angle in radians, measured from the surface normal
    eps_r   : (M, N)  relative permittivity of each layer
    d       : (M, N)  thickness of each layer in metres
    sigma   : (M, N)  electrical conductivity in S/m (default: lossless)
    mu_r    : (M, N)  relative permeability (default: 1)
    te      : (M,) bool or scalar bool. True -> TE (perpendicular) polarization,
              False -> TM (parallel) polarization.

    Returns
    -------
    S11, S21 : complex arrays of shape (M,)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    eps_r = np.atleast_2d(np.asarray(eps_r, dtype=float))

    n_samples, n_layers = eps_r.shape
    d = _as_2d(d, n_samples, n_layers, "d")
    sigma = np.zeros_like(eps_r) if sigma is None else _as_2d(
        sigma, n_samples, n_layers, "sigma")
    mu_r = np.ones_like(eps_r) if mu_r is None else _as_2d(
        mu_r, n_samples, n_layers, "mu_r")

    f = np.broadcast_to(f, (n_samples,))
    theta0 = np.broadcast_to(theta0, (n_samples,))
    te = np.broadcast_to(np.asarray(te, dtype=bool), (n_samples,))

    # --- free space on both sides -------------------------------------------
    kz0, eps0, mu0_, omega = wave_numbers(f, theta0, 1.0, 1.0, 0.0)
    z0 = wave_impedance(kz0, eps0, mu0_, omega, te)

    # --- ordered product of the layer ABCD matrices --------------------------
    A = np.ones(n_samples, dtype=complex)
    B = np.zeros(n_samples, dtype=complex)
    C = np.zeros(n_samples, dtype=complex)
    D = np.ones(n_samples, dtype=complex)

    for m in range(n_layers):
        kz_m, eps_m, mu_m, _ = wave_numbers(
            f, theta0, eps_r[:, m], mu_r[:, m], sigma[:, m])
        z_m = wave_impedance(kz_m, eps_m, mu_m, omega, te)

        phi = kz_m * d[:, m]
        a = np.cos(phi)
        b = 1j * z_m * np.sin(phi)
        c = 1j * np.sin(phi) / z_m
        dd = np.cos(phi)

        A, B, C, D = (A * a + B * c, A * b + B * dd,
                      C * a + D * c, C * b + D * dd)

    # --- ABCD -> S with equal reference impedance on both ports --------------
    denom = A + B / z0 + C * z0 + D
    s11 = (A + B / z0 - C * z0 - D) / denom
    s21 = 2.0 / denom
    return s11, s21


def single_slab_reference(f, theta0, eps_r, d, sigma=0.0, mu_r=1.0, te=True):
    """Independent closed-form check for one slab (Airy multiple-reflection sum).

    Used only in the tests: it is derived separately from the ABCD formulation,
    so agreement between the two is a meaningful verification rather than a
    restatement of the same algebra.
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))

    kz0, e0, m0, omega = wave_numbers(f, theta0, 1.0, 1.0, 0.0)
    z0 = wave_impedance(kz0, e0, m0, omega, te)

    kz1, e1, m1, _ = wave_numbers(f, theta0, eps_r, mu_r, sigma)
    z1 = wave_impedance(kz1, e1, m1, omega, te)

    r = (z1 - z0) / (z1 + z0)          # free space -> slab
    ph = np.exp(-2j * kz1 * d)         # round trip inside the slab

    gamma = r * (1.0 - ph) / (1.0 - r ** 2 * ph)
    tau = (1.0 - r ** 2) * np.exp(-1j * kz1 * d) / (1.0 - r ** 2 * ph)
    return gamma, tau
