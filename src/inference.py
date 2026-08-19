"""Predict S11 and S21 for a three-layer stack, and compare against the solver.

Examples
--------
    # surrogate prediction only
    python src/inference.py --eps 4.4 2.2 9.8 --sigma 0 0.1 0 \
                            --thickness 1.5 0.8 2.0 --freq 10 --theta 30 --pol TE

    # prediction alongside the exact solver, with the error
    python src/inference.py --eps 4.4 2.2 9.8 --thickness 1.5 0.8 2.0 \
                            --freq 10 --theta 30 --compare
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as featmod  # noqa: E402
from tmm_solver import stack_s_parameters  # noqa: E402


def load_model(outdir, feature_set):
    keras_path = os.path.join(outdir, f"surrogate_{feature_set}.keras")
    sk_path = os.path.join(outdir, f"surrogate_sklearn_{feature_set}.pkl")
    if os.path.exists(keras_path):
        from tensorflow import keras
        return keras.models.load_model(keras_path), "keras"
    if os.path.exists(sk_path):
        with open(sk_path, "rb") as fh:
            return pickle.load(fh), "sklearn"
    raise FileNotFoundError(
        f"no trained model in {outdir!r}. Run: python src/train.py --features {feature_set}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eps", type=float, nargs=3, required=True,
                   help="relative permittivity of each layer")
    p.add_argument("--sigma", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="conductivity of each layer in S/m")
    p.add_argument("--thickness", type=float, nargs=3, required=True,
                   help="layer thicknesses in mm")
    p.add_argument("--freq", type=float, required=True, help="frequency in GHz")
    p.add_argument("--theta", type=float, required=True,
                   help="incidence angle in degrees")
    p.add_argument("--pol", choices=["TE", "TM"], default="TE")
    p.add_argument("--features", choices=["raw", "physics"], default="physics")
    p.add_argument("--outdir", default="models")
    p.add_argument("--compare", action="store_true",
                   help="also evaluate the exact solver and report the error")
    args = p.parse_args()

    x_raw = np.array([[*args.eps, *args.sigma, *args.thickness,
                       args.freq, args.theta, 1.0 if args.pol == "TE" else 0.0]])

    scaler = np.load(os.path.join(args.outdir, f"scaler_{args.features}.npz"),
                     allow_pickle=True)
    x = featmod.build(x_raw, args.features)
    x = (x - scaler["lo"]) / scaler["span"]

    model, backend = load_model(args.outdir, args.features)
    pred = np.asarray(model.predict(x, verbose=0) if backend == "keras"
                      else model.predict(x))[0]
    s11_p = pred[0] + 1j * pred[1]
    s21_p = pred[2] + 1j * pred[3]

    print(f"surrogate ({backend}, '{args.features}' features)")
    print(f"  S11 = {s11_p.real:+.4f} {s11_p.imag:+.4f}j   "
          f"|S11| = {abs(s11_p):.4f}   angle = {np.degrees(np.angle(s11_p)):+.1f} deg")
    print(f"  S21 = {s21_p.real:+.4f} {s21_p.imag:+.4f}j   "
          f"|S21| = {abs(s21_p):.4f}   angle = {np.degrees(np.angle(s21_p)):+.1f} deg")

    if args.compare:
        s11_e, s21_e = stack_s_parameters(
            [args.freq * 1e9], [np.deg2rad(args.theta)],
            [args.eps], [[t * 1e-3 for t in args.thickness]],
            sigma=[args.sigma], te=(args.pol == "TE"))
        s11_e, s21_e = s11_e[0], s21_e[0]
        print("\nexact transfer-matrix solver")
        print(f"  S11 = {s11_e.real:+.4f} {s11_e.imag:+.4f}j   |S11| = {abs(s11_e):.4f}")
        print(f"  S21 = {s21_e.real:+.4f} {s21_e.imag:+.4f}j   |S21| = {abs(s21_e):.4f}")
        print("\nerror")
        print(f"  |S11 - S11_exact| = {abs(s11_p - s11_e):.4f}")
        print(f"  |S21 - S21_exact| = {abs(s21_p - s21_e):.4f}")

    metrics_path = os.path.join(args.outdir, f"metrics_{args.features}.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            m = json.load(fh)
        print(f"\n(model test-set MAE {m['overall_MAE']:.4f}, R2 {m['overall_R2']:.4f})")


if __name__ == "__main__":
    main()
