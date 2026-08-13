"""Smoke test: load a few FinText checkpoints and run one forecast each."""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from fintext_tsfm import load_chronos, load_timesfm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)

# Synthetic daily "excess return" context: AR(1) + weekly seasonality, 512 steps.
n = 512
eps = rng.normal(0, 0.01, n)
ar = np.zeros(n)
for t in range(1, n):
    ar[t] = 0.3 * ar[t - 1] + eps[t]
ctx = ar + 0.004 * np.sin(2 * np.pi * np.arange(n) / 5)

TIMESFM = [
    "FinText/TimesFM_20M_2023_Global",
    "FinText/TimesFM_20M_2023_US",
    "FinText/TimesFM_8M_2023_Global",
    "FinText/TimesFM_20M_2019_Global",
]
CHRONOS = [
    "FinText/Chronos_Small_2023_Global",
    "FinText/Chronos_Small_2023_US",
    "FinText/Chronos_Mini_2023_Global",
]

print(f"device: {DEV}\n")
print("=== TimesFM ===")
for repo in TIMESFM:
    m = load_timesfm(repo, device=DEV)
    mean, full = m.forecast([ctx], horizon=20)
    nparam = sum(p.numel() for p in m.model.parameters())
    print(f"{repo.split('/')[1]:28s} params={nparam/1e6:5.2f}M ctx={m.context_len} "
          f"H={m.horizon_len} out={tuple(full.shape)}")
    print(f"{'':28s} mean[:5]={np.round(mean[0, :5].numpy(), 5)} "
          f"q10/q90[0]={full[0,0,1]:+.5f}/{full[0,0,9]:+.5f}")

print("\n=== Chronos ===")
for repo in CHRONOS:
    p = load_chronos(repo, device=DEV)
    fc = p.predict(torch.tensor(ctx, dtype=torch.float32), prediction_length=20,
                   num_samples=100)  # [B, num_samples, H]
    q = np.quantile(fc[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    nparam = sum(x.numel() for x in p.model.parameters())
    print(f"{repo.split('/')[1]:28s} params={nparam/1e6:5.2f}M out={tuple(fc.shape)}")
    print(f"{'':28s} median[:5]={np.round(q[1][:5], 5)} q10/q90[0]={q[0][0]:+.5f}/{q[2][0]:+.5f}")

print("\nall checkpoints loaded OK")
