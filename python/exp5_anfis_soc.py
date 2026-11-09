"""
Experiment 5 - ANFIS based battery SoC prediction using Z-score data normalization
Python program (NumPy only - the ANFIS is written from first principles)

  Inputs : terminal voltage V (V), load current I (A), temperature T (deg C)
  Output : state of charge SoC (%)
  Model  : first-order Sugeno ANFIS, Gaussian membership functions, grid partition,
           hybrid learning (least squares for consequents + gradient descent for premises)

Run:  python exp5_anfis_soc.py      (bms_dataset.csv must be in this folder or its parent)
"""
import itertools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_python")
os.makedirs(OUT, exist_ok=True)

EPOCHS = 200                 # training epochs
STEP0 = 0.01                 # initial step size of the premise update
CANDIDATES = [(2, 2, 2), (3, 2, 2), (4, 2, 2), (5, 2, 2), (3, 3, 3)]   # MFs for (V, I, T)
SOC_WARN, SOC_CUTOFF = 20.0, 5.0          # low-SoC warning and load cut-off thresholds (%)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------------------
# ANFIS (first-order Sugeno, Gaussian MFs, grid partition)
# --------------------------------------------------------------------------------------
class ANFIS:
    def __init__(self, X, n_mfs):
        """Grid partition: n_mfs[i] Gaussian MFs spread evenly over the range of input i."""
        self.n_in = X.shape[1]
        self.n_mfs = list(n_mfs)
        lo, hi = X.min(axis=0), X.max(axis=0)
        self.c, self.s = [], []                       # premise parameters: centres and widths
        for i, n in enumerate(self.n_mfs):
            self.c.append(np.linspace(lo[i], hi[i], n))
            self.s.append(np.full(n, (hi[i] - lo[i]) / (n - 1) / 2.3548))   # neighbours cross at 0.5
        # one rule for every combination of membership functions
        self.rules = np.array(list(itertools.product(*[range(n) for n in self.n_mfs])))
        self.p = np.zeros((len(self.rules), self.n_in + 1))                 # consequent parameters
        self.c0 = [c.copy() for c in self.c]
        self.s0 = [s.copy() for s in self.s]

    # Layer 1 (fuzzification), layer 2 (rule firing strength), layer 3 (normalisation)
    def _layers(self, X):
        mf = [np.exp(-0.5 * ((X[:, [i]] - self.c[i]) / self.s[i]) ** 2) for i in range(self.n_in)]
        w = np.ones((len(X), len(self.rules)))
        for i in range(self.n_in):
            w *= mf[i][:, self.rules[:, i]]
        return w, w / w.sum(axis=1, keepdims=True)

    # Layer 4 (rule consequents f_r = p_r.x + r_r) and layer 5 (weighted sum)
    def predict(self, X):
        _, wn = self._layers(X)
        Xa = np.c_[X, np.ones(len(X))]
        return np.sum(wn * (Xa @ self.p.T), axis=1)

    # Forward pass of hybrid learning: consequents by least squares
    def _lse(self, X, y):
        _, wn = self._layers(X)
        Xa = np.c_[X, np.ones(len(X))]
        A = (wn[:, :, None] * Xa[:, None, :]).reshape(len(X), -1)
        self.p = np.linalg.lstsq(A, y, rcond=None)[0].reshape(self.p.shape)
        self.cond = float(np.linalg.cond(A))

    # Backward pass of hybrid learning: gradient of E = 0.5*sum(e^2) w.r.t. centres and widths
    def _gradients(self, X, y):
        w, wn = self._layers(X)
        Xa = np.c_[X, np.ones(len(X))]
        f = Xa @ self.p.T
        out = np.sum(wn * f, axis=1)
        e = out - y
        dE_dw = e[:, None] * (f - out[:, None]) / w.sum(axis=1, keepdims=True)
        gc, gs = [], []
        for i in range(self.n_in):
            gci, gsi = np.zeros(self.n_mfs[i]), np.zeros(self.n_mfs[i])
            for j in range(self.n_mfs[i]):
                use = self.rules[:, i] == j                       # rules that contain MF j of input i
                g = np.sum(dE_dw[:, use] * w[:, use], axis=1)     # dE/d(mu_ij) * mu_ij
                z = (X[:, i] - self.c[i][j]) / self.s[i][j]
                gci[j] = np.sum(g * z / self.s[i][j])
                gsi[j] = np.sum(g * z * z / self.s[i][j])
            gc.append(gci)
            gs.append(gsi)
        return gc, gs

    def fit(self, X, y, Xv, yv, epochs=EPOCHS, step=STEP0):
        """Hybrid learning. The parameters with the lowest validation RMSE are kept."""
        hist, best = [], (np.inf, None, 0)
        for ep in range(epochs):
            self._lse(X, y)
            if ep == 0:
                self.cond0 = self.cond            # conditioning with the initial grid partition
            e_tr, e_va = rmse(self.predict(X), y), rmse(self.predict(Xv), yv)
            hist.append((e_tr, e_va, step))
            if e_va < best[0]:
                best = (e_va, ([c.copy() for c in self.c], [s.copy() for s in self.s], self.p.copy()), ep + 1)
            gc, gs = self._gradients(X, y)
            norm = np.sqrt(sum(np.sum(g ** 2) for g in gc + gs)) + 1e-12
            for i in range(self.n_in):                              # step of length `step` along -gradient
                self.c[i] -= step * gc[i] / norm
                self.s[i] = np.maximum(self.s[i] - step * gs[i] / norm, 1e-3)
            e = [h[0] for h in hist]                                # step-size adaptation (Jang, 1993)
            if len(e) >= 5 and all(e[-k] < e[-k - 1] for k in range(1, 5)):
                step *= 1.1
            if len(e) >= 5 and e[-1] > e[-2] < e[-3] > e[-4] < e[-5]:
                step *= 0.9
        (self.c, self.s, self.p), self.best_epoch = best[1], best[2]
        self.hist = np.array(hist)
        return self


# --------------------------------------------------------------------------------------
# 1. Data, split and Z-score normalisation
# --------------------------------------------------------------------------------------
csv = next(p for p in (os.path.join(HERE, "bms_dataset.csv"), os.path.join(HERE, "..", "bms_dataset.csv"))
           if os.path.exists(p))
data = np.loadtxt(csv, delimiter=",", skiprows=1)
X, y = data[:, :3], data[:, 3]
names = ["Voltage (V)", "Current (A)", "Temperature (°C)"]

k = np.arange(len(y)) % 20                     # systematic split: of every 20 rows, 14 / 3 / 3
tr, va, te = k < 14, (k >= 14) & (k < 17), k >= 17
mu, sigma = X[tr].mean(axis=0), X[tr].std(axis=0, ddof=1)      # from the TRAINING set only
Z = (X - mu) / sigma
print(f"Samples: train {tr.sum()}, validation {va.sum()}, test {te.sum()}")
print("Z-score parameters   mu =", np.round(mu, 4), "  sigma =", np.round(sigma, 4))

# --------------------------------------------------------------------------------------
# 2. Structure selection on the validation set, then the reference models
# --------------------------------------------------------------------------------------
print("\nStructure selection (Z-score inputs)")
trials = {}
for n_mfs in CANDIDATES:
    m = ANFIS(Z[tr], n_mfs).fit(Z[tr], y[tr], Z[va], y[va])
    trials[n_mfs] = m
    print(f"  MFs {n_mfs}  rules {len(m.rules):2d}  train {rmse(m.predict(Z[tr]), y[tr]):.4f}"
          f"  validation {rmse(m.predict(Z[va]), y[va]):.4f}  test {rmse(m.predict(Z[te]), y[te]):.4f}")
best_mfs = min(trials, key=lambda n: rmse(trials[n].predict(Z[va]), y[va]))
anfis = trials[best_mfs]
print(f"Selected structure: {best_mfs} -> {len(anfis.rules)} rules (best epoch {anfis.best_epoch})")

A = np.c_[Z[tr], np.ones(tr.sum())]                                     # linear regression baseline
beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
raw = ANFIS(X[tr], best_mfs).fit(X[tr], y[tr], X[va], y[va])             # same ANFIS, no normalisation
v_only = ANFIS(Z[tr][:, :1], best_mfs[:1]).fit(Z[tr][:, :1], y[tr], Z[va][:, :1], y[va])   # voltage only

pred = {"Linear regression": np.c_[Z[te], np.ones(te.sum())] @ beta,
        "ANFIS, voltage only": v_only.predict(Z[te][:, :1]),
        "ANFIS, raw inputs": raw.predict(X[te]),
        "ANFIS, Z-score inputs": anfis.predict(Z[te])}


def metrics(p, t):
    e = p - t
    return {"RMSE": rmse(p, t), "MAE": float(np.mean(np.abs(e))), "MaxErr": float(np.max(np.abs(e))),
            "R2": float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2))}


results = {name: metrics(p, y[te]) for name, p in pred.items()}
print("\nTest-set performance (150 unseen samples)")
for name, r in results.items():
    print(f"  {name:24s} RMSE {r['RMSE']:.4f} %  MAE {r['MAE']:.4f} %  max {r['MaxErr']:.4f} %  R2 {r['R2']:.6f}")
print(f"Condition number of the least-squares matrix at epoch 1: Z-score {anfis.cond0:.2e}, raw {raw.cond0:.2e}")


# --------------------------------------------------------------------------------------
# 3. Real-time BMS routine: one call per sampling instant
# --------------------------------------------------------------------------------------
def bms_step(v, i, t):
    """Raw sensor reading -> Z-score -> ANFIS -> 0..100 % limit -> warning / cut-off flags."""
    z = (np.array([[v, i, t]]) - mu) / sigma
    soc = float(np.clip(anfis.predict(z)[0], 0.0, 100.0))
    return soc, soc < SOC_WARN, soc < SOC_CUTOFF


order = np.argsort(-y[te])                      # replay the test samples from full to empty
Xs, ys = X[te][order], y[te][order]
live = np.array([bms_step(*row) for row in Xs])
print(f"\nBMS replay of {len(ys)} test samples: RMSE {rmse(live[:, 0], ys):.4f} %,"
      f" warning flag on {int(live[:, 1].sum())} samples, cut-off flag on {int(live[:, 2].sum())} samples")
print("   V (V)    I (A)   T (C)   SoC actual  SoC ANFIS  warn cut")
for j in np.round(np.linspace(0, len(ys) - 1, 8)).astype(int):
    print(f"  {Xs[j, 0]:6.3f}  {Xs[j, 1]:7.3f}  {Xs[j, 2]:6.2f}   {ys[j]:8.3f}   {live[j, 0]:8.3f}"
          f"    {int(live[j, 1])}    {int(live[j, 2])}")

# --------------------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------------------
# (a) training and validation error
fig, ax = plt.subplots(figsize=(6.8, 3.6))
ep = np.arange(1, EPOCHS + 1)
ax.semilogy(ep, anfis.hist[:, 0], color=BLUE, lw=2, label="Z-score inputs, training")
ax.semilogy(ep, anfis.hist[:, 1], color=BLUE, lw=2, ls="--", label="Z-score inputs, validation")
ax.semilogy(ep, raw.hist[:, 0], color=ORANGE, lw=2, label="Raw inputs, training")
ax.semilogy(ep, raw.hist[:, 1], color=ORANGE, lw=2, ls="--", label="Raw inputs, validation")
ax.set(xlabel="Epoch", ylabel="RMSE (% SoC)", title=f"Python ANFIS {list(best_mfs)}: hybrid-learning error")
ax.legend(frameon=False, fontsize=9)
fig.savefig(os.path.join(OUT, "py_error_curves.png"))

# (b) membership functions before and after training
fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
for i, ax in enumerate(axes):
    zz = np.linspace(Z[tr][:, i].min(), Z[tr][:, i].max(), 300)
    for j in range(best_mfs[i]):
        ax.plot(zz, np.exp(-0.5 * ((zz - anfis.c0[i][j]) / anfis.s0[i][j]) ** 2), color=GREY, lw=1.2, ls=":",
                label="Initial" if j == 0 else None)
        ax.plot(zz, np.exp(-0.5 * ((zz - anfis.c[i][j]) / anfis.s[i][j]) ** 2), color=BLUE, lw=2,
                label="Trained" if j == 0 else None)
    ax.set(xlabel=f"z-score of {names[i].split(' (')[0].lower()}", ylim=(0, 1.05))
axes[0].set_ylabel("Degree of membership")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Python ANFIS: membership functions before and after training", y=1.02)
fig.savefig(os.path.join(OUT, "py_membership_functions.png"))

# (c) predicted against actual SoC, and the error of every test sample
p = pred["ANFIS, Z-score inputs"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot([0, 100], [0, 100], color=GREY, lw=1, label="Ideal (y = x)")
axes[0].scatter(y[te], p, s=22, color=BLUE, edgecolor="white", linewidth=0.5, label="Test samples", zorder=3)
axes[0].set(xlabel="Actual SoC (%)", ylabel="Predicted SoC (%)", title="Predicted against actual SoC")
axes[0].legend(frameon=False)
axes[1].axhline(0, color=GREY, lw=1)
axes[1].scatter(y[te], 1000 * (p - y[te]), s=22, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)
axes[1].set(xlabel="Actual SoC (%)", ylabel="Prediction error (×10$^{-3}$ % SoC)",
            title=f"Test error (RMSE = {results['ANFIS, Z-score inputs']['RMSE']:.4f} %)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "py_predicted_vs_actual.png"))

with open(os.path.join(HERE, "python_results.json"), "w") as fh:
    json.dump({"n": [int(tr.sum()), int(va.sum()), int(te.sum())], "mu": mu.tolist(), "sigma": sigma.tolist(),
               "selection": {str(n): {"rules": int(len(m.rules)), "train": rmse(m.predict(Z[tr]), y[tr]),
                                      "val": rmse(m.predict(Z[va]), y[va]), "test": rmse(m.predict(Z[te]), y[te]),
                                      "best_epoch": int(m.best_epoch)} for n, m in trials.items()},
               "selected": list(best_mfs), "results": results, "cond_z": anfis.cond0, "cond_raw": raw.cond0,
               "raw_best_epoch": int(raw.best_epoch), "raw_hist": raw.hist[:, :2].tolist(), "hist": anfis.hist[:, :2].tolist(),
               "centres": [c.tolist() for c in anfis.c], "widths": [s.tolist() for s in anfis.s],
               "replay": {"rmse": rmse(live[:, 0], ys), "warn": int(live[:, 1].sum()), "cut": int(live[:, 2].sum())}},
              fh, indent=1)
print(f"\nFigures saved in {OUT}")
