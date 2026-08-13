"""Show what the two model families actually emit, before any post-processing."""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ebenezer0616/IPO_test/scripts")
from fintext_tsfm import load_chronos, load_timesfm

np.set_printoptions(precision=4, suppress=True, linewidth=160)
DEV = "cuda"

d = np.load("/home/ebenezer0616/IPO_test/data/tw_ipo_panel.npz", allow_pickle=True)
i = int(np.where(d["ipo_year"] == 2024)[0][0])
ctx = d["ctx_excess"][i].astype(np.float32)
print(f"context: stock {d['stock_id'][i]}, 30 daily excess returns, "
      f"mean {ctx.mean():+.4f} std {ctx.std():.4f}\n")

# ===================== TimesFM =====================
print("=" * 78)
print("TimesFM  (patched decoder, direct multi-step)")
print("=" * 78)

m = load_timesfm("FinText/TimesFM_20M_2023_Global", device=DEV)
mdl, C = m.model, m.context_len
freq = torch.zeros(1, 1, dtype=torch.long, device=DEV)

for n_real in (30, 128):
    inp = torch.zeros(1, C); pad_in = torch.ones(1, C)
    s = np.resize(ctx, n_real)                            # 128-day case just tiles the context
    inp[0, C - n_real:] = torch.tensor(s); pad_in[0, C - n_real:] = 0.0
    inp, pad_in = inp.to(DEV), pad_in.to(DEV)

    with torch.no_grad():
        _, ppad, stats, _ = mdl._preprocess_input(input_ts=inp, input_padding=pad_in)
        raw = mdl(inp, pad_in, freq)                      # [1, n_patches, 128, 10]

    last = raw[0, -1].cpu().numpy()                       # [128, 10]
    q = last[:, 1:]
    print(f"\n--- context = {n_real} days ---")
    print(f"raw forward() output: {tuple(raw.shape)}  "
          f"= (batch, n_patches=512/32, horizon_len=128, 1 point + 9 quantiles)")
    print(f"valid (non-padded) patches: {int((ppad[0] == 0).sum())}/16   "
          f"context normalisation: mu={stats[0].item():+.6f} sigma={stats[1].item():.6f}")
    print("last patch, horizon steps 1-3, all 10 channels "
          "[0]=point, [1..9]=quantiles q10..q90:")
    print(last[:3])
    print(f"point head : mean {last[:,0].mean():+.5f}  std {last[:,0].std():.5f}  "
          f"max|.| {np.abs(last[:,0]).max():.5f}")
    print(f"quantiles  : max|.| {np.abs(q).max():8.3f}   "
          f"monotone q10<=...<=q90 in {(np.diff(q, axis=1) >= 0).all(1).mean():.0%} of steps")
del m
torch.cuda.empty_cache()

# ===================== Chronos =====================
print("\n" + "=" * 78)
print("Chronos  (T5 over quantised return tokens, autoregressive sampling)")
print("=" * 78)

p = load_chronos("FinText/Chronos_Small_2023_Global", device=DEV)
tok, cc = p.tokenizer, p.model.config
inner = p.model.model                                    # T5ForConditionalGeneration
ids, attn, scale = tok.context_input_transform(torch.tensor(ctx).unsqueeze(0))
print(f"\ncontext_input_transform -> token ids {tuple(ids.shape)}, mean-scale = {scale.item():.6f}")
print("token ids (last 12 of the context, then EOS):", ids[0, -12:].tolist())
print(f"vocab: {cc.n_tokens} tokens = {cc.n_special_tokens} special + {len(tok.centers)} uniform "
      f"bins over [{cc.tokenizer_kwargs['low_limit']}, {cc.tokenizer_kwargs['high_limit']}] "
      f"in mean-scaled units (1 unit = {scale.item():.6f} of raw return here)")

from transformers import GenerationConfig
with torch.no_grad():
    out = inner.generate(input_ids=ids.to(DEV), attention_mask=attn.to(DEV),
                         generation_config=GenerationConfig(
                             min_new_tokens=30, max_new_tokens=30, do_sample=True,
                             num_return_sequences=8, top_k=50, temperature=1.0,
                             eos_token_id=None, pad_token_id=0))
print(f"\nmodel.generate -> {tuple(out.shape)} raw token ids (8 sampled paths x 31)")
print("sample path 0 token ids:", out[0, 1:9].tolist())
vals = tok.output_transform(out[:, 1:].unsqueeze(0).cpu(), scale)   # [1, 8, 30]
print(f"output_transform -> {tuple(vals.shape)} daily excess returns")
print("path 0, first 8 days:", vals[0, 0, :8].numpy())
v = vals[0].numpy()
print(f"\nacross the 8 paths, day 1: min {v[:,0].min():+.4f} median {np.median(v[:,0]):+.4f} "
      f"max {v[:,0].max():+.4f}")
print("ChronosPipeline.predict() just wraps this and returns the sample tensor "
      "[batch, num_samples, horizon]; the point forecast is whatever statistic you take.")
