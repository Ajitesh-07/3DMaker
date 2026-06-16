# Bicycle (mip-NeRF 360) — SOTA PSNR Reference

Published per-scene **PSNR on the `bicycle` scene** of the mip-NeRF 360 dataset, so we have a
yardstick for our own engine. Researched June 2026; numbers cross-checked across the original
papers and the reproducible NerfBaselines eval.

## The numbers (standard protocol)

| Method | Bicycle PSNR ↑ | Class | Notes |
|---|---|---|---|
| Plenoxels | **21.91** | sparse voxel grid (no MLP) | floor of the modern methods |
| Instant-NGP | **~22.2** (22.17–22.79) | hash grid + occupancy | **closest to our engine** |
| Nerfacto | **24.08** | nerfstudio (INGP + mip-360 ideas) | strong hybrid baseline |
| Mip-NeRF 360 | **24.3–24.4** (24.31 / 24.40) | the original 360 method | |
| 3D Gaussian Splatting (3DGS) | **~25.2** (25.03–25.63) | rasterized splats | real-time |
| Mip-Splatting | **~25.5** (25.13–25.72) | anti-aliased 3DGS | |
| **Zip-NeRF** | **25.80** | anti-aliased grid NeRF | **current SOTA-class on bicycle** |

So the modern spread on bicycle is roughly **22 → 26 PSNR**, with **Zip-NeRF ≈ 25.8** at the top
and **Instant-NGP ≈ 22** at the bottom of the "good" methods (Plenoxels trailing at ~21.9).

## Protocol (read before comparing)

These are **held-out TEST PSNR**, not training PSNR:
- **Train/test split:** every **8th** image is held out for test; the rest train. PSNR is on the
  held-out views.
- **Resolution:** mip-NeRF 360 *outdoor* scenes (bicycle, garden, stump, flowers, treehill) are
  evaluated **downsampled 4×** (`images_4/`, ≈1237×822). Indoor scenes use 2×. Reporting at full
  resolution gives *different* (often higher) numbers — that's why some papers quote bicycle at
  27–28; those are not the standard 4× protocol.
- **Cross-paper variance** of ±0.3–0.6 PSNR is normal (COLMAP poses, masking, eval code differ).

## Where our engine fits

**Our engine is Instant-NGP-class** (multiresolution hash grid + cascaded occupancy grid + fused
MLP), so the honest target is **~22 test PSNR — i.e. parity with Instant-NGP.** Reaching that from
a *from-scratch* CUDA implementation is a legitimate, competitive result.

**Important caveat on our current 22.0:** that is **TRAINING PSNR** (computed on the training rays
inside the step), **not** the held-out test PSNR the table above reports. Train PSNR is typically
**higher** than test PSNR, so our true comparable number is likely **~19–21**, in/just under
Instant-NGP territory — exactly where an INGP-class engine should sit, with the background being
the thing holding it back (the green-shell smear at oblique views).

To get a number that's *directly* comparable to the table, we need to:
1. Hold out every 8th image (don't train on it).
2. Render those held-out views and compute PSNR vs. their ground truth (our `train_hit` currently
   evaluates on *train* rays — a real test-set eval is a small addition).
3. Render at the same 4× resolution.

## The gap to SOTA, and what closes it

The ~22 → ~25.8 gap (Instant-NGP → Zip-NeRF) on bicycle is almost entirely:
- **Background / unbounded handling** — Zip-NeRF's sample placement + contraction resolve the far
  field; our outer cascade is a coarse shell (our current frontier: more cascades, distortion in
  normalized `s`).
- **Anti-aliasing** — Zip-NeRF/Mip-Splatting integrate over the pixel cone (multisampling / scale
  features); we point-sample. This is a big chunk of the outdoor-scene gap.
- **Sample density / quadrature** — sub-voxel sampling (already wired as `samplesPerVoxel`) is our
  lever here.

So "match Instant-NGP (~22)" is the near goal; "approach Zip-NeRF (~26)" needs anti-aliasing +
better unbounded sampling, which is research-grade work beyond parity.

## Sources

- 3D Gaussian Splatting: <https://arxiv.org/pdf/2308.04079>
- Mip-Splatting (per-scene 3DGS/Mip-Splatting table): <https://arxiv.org/pdf/2311.16493>
- Zip-NeRF per-scene (paper values via reference impl): <https://github.com/SuLvXiangXin/zipnerf-pytorch>
- NerfBaselines (reproducible per-scene eval, incl. Plenoxels/Nerfacto/INGP): <https://arxiv.org/pdf/2406.17345>
- Nerfstudio / Nerfacto: <https://arxiv.org/pdf/2302.04264>
