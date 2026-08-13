"""Loaders for FinText TSFM checkpoints (https://huggingface.co/FinText).

Two families, each published as {size} x {year 2000..2023} x {Global, US, Augmented}:

  TimesFM_{8M,20M}_{year}_{variant}    -> timesfm 1.x PyTorch PatchedTimeSeriesDecoder
                                          (config.json says architectures=["TimesFMForHF"],
                                           which is NOT a transformers class; weights are the
                                           original google-research/timesfm torch names with a
                                           "model." prefix)
  Chronos_{Tiny,Mini,Small}_{year}_{variant} -> chronos-forecasting T5 pipeline

Both are pre-trained on *excess returns* (1990 -> {year}), so feed them return series,
not raw prices. The year in the repo id is the training cut-off: to evaluate IPOs in
year Y use a checkpoint trained up to Y-1 if you want a point-in-time-clean test.
"""

import importlib.util
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# timesfm's package __init__ drags in jax/wandb; the torch decoder file is standalone.
_PPD_PATH = os.path.join(
    os.path.dirname(importlib.util.find_spec("timesfm").origin),
    "pytorch_patched_decoder.py",
)
_spec = importlib.util.spec_from_file_location("_ppd", _PPD_PATH)
_ppd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ppd)

TimesFMConfig = _ppd.TimesFMConfig
PatchedTimeSeriesDecoder = _ppd.PatchedTimeSeriesDecoder

FREQ_HIGH = 0  # timesfm freq bucket for high-frequency (daily/hourly) series


class FinTextTimesFM:
    """Thin wrapper: context (returns) in, quantile forecast out."""

    def __init__(self, model, config, repo_id):
        self.model = model
        self.config = config
        self.repo_id = repo_id
        self.context_len = config.context_len
        self.horizon_len = config.horizon_len
        self.patch_len = config.patch_len
        self.quantiles = config.quantiles

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forecast(self, series, horizon=None, freq=FREQ_HIGH, pad_to_full=False):
        """series: list of 1-D sequences (ragged ok). Returns (mean, quantiles).

        mean:      [B, horizon]
        quantiles: [B, horizon, 1 + len(quantiles)]  (index 0 is the mean head)

        pad_to_full: pad every series out to the full 512-step context. Measurably worse
          for short inputs (a 30-step context padded to 512 roughly doubles the predicted
          magnitude and costs ~6pp of out-of-sample R^2), so the default only pads up to
          the next multiple of patch_len, which is all the patching requires.
        """
        horizon = horizon or self.horizon_len
        dev = self.device
        longest = max(len(s) for s in series)
        ctx = self.context_len if pad_to_full else \
            min(self.context_len, -(-longest // self.patch_len) * self.patch_len)
        b = len(series)

        inputs = torch.full((b, ctx), 0.0, dtype=torch.float32)
        paddings = torch.ones((b, ctx + horizon), dtype=torch.float32)
        for i, s in enumerate(series):
            s = torch.as_tensor(s, dtype=torch.float32)[-ctx:]
            n = s.shape[0]
            inputs[i, ctx - n:] = s
            paddings[i, ctx - n:ctx] = 0.0  # 0 = real observation, 1 = padding

        mean, full = self.model.decode(
            input_ts=inputs.to(dev),
            paddings=paddings.to(dev),
            freq=torch.full((b, 1), freq, dtype=torch.long, device=dev),
            horizon_len=horizon,
        )
        return mean.float().cpu(), full.float().cpu()


def load_timesfm(repo_id, device="cuda", use_positional_embedding=True, strict=True,
                 fix_scaling=True):
    """repo_id e.g. 'FinText/TimesFM_20M_2023_Global' (or a local snapshot dir).

    fix_scaling: the published checkpoints carry uninitialised memory in every
      `self_attn.scaling` (timesfm's per_dim_scale) -- values from 1e-32 to 1e38 with
      repeated byte patterns, i.e. `torch.empty` that was never written. The forward pass
      computes `q * 1.442695/sqrt(head_dim) * softplus(scaling)`, so garbage there
      destroys attention. softplus(0) = ln2 = 1/1.442695 exactly, so setting it to 0
      recovers standard 1/sqrt(head_dim) scaling. Set False to load the raw values.
    """
    path = repo_id if os.path.isdir(repo_id) else snapshot_download(repo_id)
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)

    cfg = TimesFMConfig(
        num_layers=raw["num_layers"],
        num_heads=raw["num_heads"],
        num_kv_heads=raw["num_kv_heads"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        head_dim=raw["head_dim"],
        patch_len=32,                      # implied by input_ff_layer in_features = 2*32
        horizon_len=raw["horizon_len"],
        dtype="float32",
        use_positional_embedding=use_positional_embedding,
    )
    cfg.context_len = raw["context_len"]    # not a TimesFMConfig field; kept for callers

    model = PatchedTimeSeriesDecoder(cfg)
    sd = load_file(os.path.join(path, "model.safetensors"))
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    if fix_scaling:
        for k in sd:
            if k.endswith("self_attn.scaling"):
                sd[k] = torch.zeros_like(sd[k])
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")

    model.eval().to(device)
    return FinTextTimesFM(model, cfg, repo_id)


def load_chronos(repo_id, device="cuda", dtype=torch.float32):
    """repo_id e.g. 'FinText/Chronos_Small_2023_Global'. Returns a ChronosPipeline."""
    from chronos import ChronosPipeline

    return ChronosPipeline.from_pretrained(repo_id, device_map=device, torch_dtype=dtype)
