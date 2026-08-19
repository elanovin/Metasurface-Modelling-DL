"""Train the S-parameter surrogate network.

Usage
-----
    python src/train.py --data data/dataset.npz --features physics
    python src/train.py --data data/dataset.npz --features raw     # ablation

The network is a fully connected regressor mapping the physical description of
a three-layer stack to the real and imaginary parts of S11 and S21. Keras is
used when TensorFlow is available; otherwise the script falls back to a
scikit-learn MLP so that the repository is runnable without a deep learning
install.

Evaluation reports both the regression error and a physical diagnostic: the
fraction of predictions that violate passivity, |S11|^2 + |S21|^2 <= 1. The
network is not constrained to respect that bound, so it is a meaningful check
of whether the model has learned physically consistent behaviour.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as featmod  # noqa: E402

HIDDEN = [512, 256, 128, 64]
DROPOUT = [0.4, 0.3, 0.2, 0.0]


def split(X, y, seed=0, val_frac=0.1, test_frac=0.1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    n_val = int(len(X) * val_frac)
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def minmax_fit(X):
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return lo, span


def evaluate(y_true, y_pred, target_names):
    out = {"overall_MAE": float(np.mean(np.abs(y_true - y_pred))),
           "overall_RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))}

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    out["overall_R2"] = float(1.0 - ss_res / ss_tot)

    for i, name in enumerate(target_names):
        out[f"MAE_{name}"] = float(np.mean(np.abs(y_true[:, i] - y_pred[:, i])))

    s11_t, s21_t = y_true[:, 0] + 1j * y_true[:, 1], y_true[:, 2] + 1j * y_true[:, 3]
    s11_p, s21_p = y_pred[:, 0] + 1j * y_pred[:, 1], y_pred[:, 2] + 1j * y_pred[:, 3]
    out["MAE_abs_S11"] = float(np.mean(np.abs(np.abs(s11_t) - np.abs(s11_p))))
    out["MAE_abs_S21"] = float(np.mean(np.abs(np.abs(s21_t) - np.abs(s21_p))))

    energy = np.abs(s11_p) ** 2 + np.abs(s21_p) ** 2
    out["passivity_violation_rate"] = float(np.mean(energy > 1.0 + 1e-3))
    out["max_predicted_energy"] = float(energy.max())
    return out


def train_keras(Xtr, ytr, Xva, yva, epochs, batch_size, outdir, tag):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([keras.Input(shape=(Xtr.shape[1],))])
    for units, drop in zip(HIDDEN, DROPOUT):
        model.add(layers.Dense(units, activation="relu",
                               kernel_initializer="he_normal"))
        model.add(layers.BatchNormalization())
        if drop > 0:
            model.add(layers.Dropout(drop))
    model.add(layers.Dense(ytr.shape[1], activation="linear"))

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
              batch_size=batch_size, callbacks=callbacks, verbose=2)

    model.save(os.path.join(outdir, f"surrogate_{tag}.keras"))
    return model.predict(Xva, verbose=0), model


def train_sklearn(Xtr, ytr, Xva, yva, epochs, batch_size, outdir, tag):
    import pickle
    import warnings

    from sklearn.neural_network import MLPRegressor

    warnings.filterwarnings("ignore")
    model = MLPRegressor(hidden_layer_sizes=tuple(HIDDEN), activation="relu",
                         batch_size=batch_size, learning_rate_init=1e-3,
                         max_iter=epochs, early_stopping=True,
                         n_iter_no_change=10, random_state=0, verbose=True)
    model.fit(Xtr, ytr)
    with open(os.path.join(outdir, f"surrogate_sklearn_{tag}.pkl"), "wb") as fh:
        pickle.dump(model, fh)
    return model.predict(Xva), model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--features", choices=["raw", "physics"], default="physics")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--outdir", default="models")
    args = p.parse_args()

    blob = np.load(args.data, allow_pickle=True)
    X_raw, y = blob["X"], blob["y"]
    target_names = list(blob["target_names"])

    X = featmod.build(X_raw, args.features)
    feat_names = featmod.names(blob["feature_names"], args.features)
    print(f"feature set '{args.features}': {X.shape[1]} features, {len(X):,} samples")

    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    lo, span = minmax_fit(Xtr)
    Xtr, Xva, Xte = ((a - lo) / span for a in (Xtr, Xva, Xte))

    os.makedirs(args.outdir, exist_ok=True)
    try:
        import tensorflow  # noqa: F401
        backend = "keras"
    except ImportError:
        backend = "sklearn"
        print("TensorFlow not found - falling back to the scikit-learn MLP.")

    trainer = train_keras if backend == "keras" else train_sklearn
    _, model = trainer(Xtr, ytr, Xva, yva, args.epochs, args.batch_size,
                       args.outdir, args.features)

    y_pred = model.predict(Xte) if backend == "sklearn" else model.predict(Xte, verbose=0)
    metrics = evaluate(yte, np.asarray(y_pred), target_names)
    metrics.update({"backend": backend, "feature_set": args.features,
                    "n_features": int(X.shape[1]), "n_samples": int(len(X)),
                    "n_test": int(len(Xte))})

    print("\ntest set results")
    for k, v in metrics.items():
        print(f"  {k:28s} {v}")

    np.savez(os.path.join(args.outdir, f"scaler_{args.features}.npz"),
             lo=lo, span=span, feature_names=np.array(feat_names))
    with open(os.path.join(args.outdir, f"metrics_{args.features}.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
