# 3DMaker — Code Review & Roadmap

A consolidated reference covering: the NaN training fix, a full code review of both
modules, the plan for real-world unbounded scenes, the 3DGS question, and the
end-goal of clean mesh export to FBX.

---

## Table of Contents

1. [The NaN Training Bug (fixed)](#1-the-nan-training-bug-fixed)
2. [Code Review](#2-code-review)
3. [Real-World Unbounded Scenes](#3-real-world-unbounded-scenes)
4. [NeRF vs 3D Gaussian Splatting](#4-nerf-vs-3d-gaussian-splatting)
5. [The End Goal: Clean Mesh Export to FBX](#5-the-end-goal-clean-mesh-export-to-fbx)
6. [Recommended Sequencing](#6-recommended-sequencing)
7. [Reading List (Master)](#7-reading-list-master)

---

## 1. The NaN Training Bug (fixed)

### Symptom
On real-world scenes (not synthetic), PSNR climbs nicely, then loss goes **NaN and
never recovers**.

### Root cause — three compounding problems

1. **Adam optimizers had no NaN/Inf guard (the reason it never recovers).**
   In `optimizerKernel.cu`, all three updaters (`fusedAdamWeightsOptim`,
   `fusedAdamBiasOptim`, `fusedAdamHashGridOptim`) updated `m`, `v`, and the weights
   with no check that the gradient was finite. The per-sample gradients in
   `compute_color_grad` / `compute_density_grad` *are* clamped and NaN-scrubbed, but
   the **weight/bias/hashgrid gradients** from the WMMA backward kernels are not. A
   single `Inf` latches `m`/`v` to `Inf`/`NaN` **permanently** → every later weight
   becomes `NaN`. This is why the failure is permanent, not a recoverable spike.

2. **fp16 activation overflow is the trigger — much likelier on real scenes.**
   Forward kernels store hidden activations as `__float2half(val)` with **no clamp**
   (`networkFusionHashtable.cu:309`, last-layer output `:307`), and `compute_SH_gather`
   packs the 16 raw density logits straight into fp16 as the color-MLP input. fp16
   maxes at 65504. On **synthetic** the object sits in a mostly-empty AABB; on
   **real-world** the whole `[-1.5,1.5]³` box is filled, the hash grid learns much
   larger features, and once PSNR is high those activations overflow to `inf` → feeds
   `inf` into the unguarded Adam → permanent NaN.

3. **Cosine-annealing LR was a no-op pinned at 1e-2 (amplifier).**
   In `Pipeline.cu`, `max_lr = 1e-2` **and** `min_lr = 1e-2`, so the schedule never
   decayed — LR sat high through all training, including late stages where gradients
   spike (exactly when the NaN hit).

### Fixes applied

| File | Change | Purpose |
|------|--------|---------|
| `optimizerKernel.cu` | NaN/Inf guards + self-heal in all 3 Adam kernels (`finite_or_zero` + final `isfinite` guard) | **Stops the permanent NaN** — bad gradients can no longer latch into `m`/`v`/weights |
| `otherKernels.cu` (`compute_SH_gather`) | Sanitize the 16 density logits before `expf` / color-input packing | Breaks the density→color `inf`/NaN propagation path at the source |
| `Pipeline.cu` | `min_lr` 1e-2 → 1e-4 | Cosine LR now actually decays, stabilizing late training |

Fixes #1/#2 are pure safety (no behavioral change on healthy steps); #3 restores
intended LR decay. Build verified (Release, exit 0).

### Optional follow-on hardening (not yet done)
- Clamp fp16 hidden activations directly at the `__float2half` store sites
  (`networkFusion.cu:188,328`) to stop transient loss spikes at the source.
- Add a per-step host-side NaN check on the loss (currently only computed every 10
  chunks in `Pipeline.cu`).

---

## 2. Code Review

### 2.1 Structure

A clean two-layer split with a thin orchestrator on top:

```
src/                    main.cpp (CLI) + render.cpp → 3DMaker.exe / 3DViewer.exe
modules/NeRF/           Scene + rendering: InstantNerf, Pipeline, processRays,
                        DataLoader, compositing, otherKernels, RenderKernels
modules/TinyMLP/        Math engine: TinyMLP (color), TinyMLPHashGrid (density),
                        fused WMMA forward/backward, fused Adam, hashgrid encode
scripts/colmap2nerf.py  Preprocessing: video → frames → COLMAP → transforms.json
```

The separation is genuinely good: **TinyMLP knows nothing about NeRF** (a generic
fused tensor-core MLP + hash-grid engine with its own benchmarks); **NeRF orchestrates
it**. `InstantNerf` owns the occupancy grid and the train/render loops; `Pipeline.cu`
is the driver; two memory modes (`TRAINING`/`INFERENCE`) trade VRAM for activation
storage.

**Structural weak points:**
- **Three near-identical DDA marchers** in `processRays.cu` (`:57`, `:563`, `:693`) —
  any sampling change must be made in three places.
- **`mlpKernel.cu` (~550 lines) is dead code** — the single-layer family and
  `launchForwardKernel` are marked "legacy dont use" (`TinyMLP.h:157`).
- Save format keys on **padded** dims (`hw_opt`), so different requested dims that pad
  to the same size silently cross-load.

### 2.2 What it currently has (strong)

A legitimately strong from-scratch implementation, not a toy:
- **Fully-fused tensor-core MLP**: activations stay resident in shared memory across
  all layers (`networkFusion.cu`), `cp.async` 128-bit weight prefetch with 2-deep
  pipelining, `PAD=8` to kill bank conflicts, fp32 WMMA accumulators,
  `__launch_bounds__(256,3)`. Correctly done.
- **fp32 master weights + fp16 forward copies + fused Adam** (standard mixed-precision).
- **Multi-resolution hash grid** with dense-vs-hashed level split.
- **Hierarchical occupancy bitgrid** with mipmap max-pooling, EMA density updates, and
  a coarse-to-fine DDA marcher that skips empty space.
- **Hit-centric compaction** + Morton sorting of samples for hash-grid cache locality.
- Plus the NaN-hardening added above.

The README's headline claims (100M+ hash evals/s, fused Adam, dynamic warp folding) are
backed by real code.

### 2.3 Is it optimized?

**Mostly yes at the kernel level**, with specific pipeline-level inefficiencies.

Left on the table (by likely impact):
1. **Redundant host↔device syncs on the hot path.** `processRaysHitLinear` does
   **three** separate `cudaMemcpyAsync` + `cudaStreamSynchronize` per chunk
   (`processRays.cu:855-889`). Probably the biggest throughput leak.
2. **CUB temp storage allocated/freed every call** (`processRays.cu:351-404`) instead
   of reused.
3. **Mapped-memory image reads over PCIe per ray** (`DataLoader.cu:203-206`);
   randomly-shuffled rays make each warp scatter across the whole dataset —
   uncoalesced PCIe traffic every step. A device-resident texture would be far faster.
   Despite the name it does **not** stream from disk; the whole dataset must fit in
   pinned host RAM.
4. **Backward kernel lacks `__launch_bounds__`** (`networkFusionBackward.cu`).
5. **Single grid dimension (batch only)** — under-fills the GPU for small batches/dims.
6. Minor: `get_mipmap_offset` recomputed in the marcher loop; dead
   `laneId = threadIdx.x + 1 - 1` no-ops.

### 2.4 Correctness / robustness risks (TinyMLP)

- **Silent no-ops on unsupported config** (highest-impact latent bug): every launcher
  `default: break`s with no error if `maxDim ∉ {16,32,64,128}`, `maxDim>128`, or
  `numLayers>10` (`networkFusion.cu:421`, backward `:663`), leaving output buffers
  stale. Add an assert/log.
- **No `cudaGetLastError()` after kernel launches** anywhere in TinyMLP — launch
  failures are invisible until an unrelated later sync.
- **HashGrid backward depends on a cached raw input pointer**
  (`d_current_inputs_backward`, `TinyMLPHashGrid.cu:258`) with no null guard —
  `backward` before `forward` dereferences null.
- Weight init is **uniform LeCun** (`±1/√fan_in`), not the labeled "Kaiming" — missing
  the He √2 gain for ReLU (minor; comment is misleading).

### 2.5 Hard-coded assumptions (quick reference)

| Assumption | Location |
|---|---|
| AABB hardcoded `[-1.5,1.5]³`, never data-derived | `Pipeline.cu:111-112` |
| `MAX_HITS = 128` per ray (silent truncation) | `InstantNerf.h:23`; loop `processRays.cu:97` |
| `maxDim ∈ {16,32,64,128}`, `≤128`, else silent no-op | `networkFusion.cu:389,421` |
| `numLayers ≤ 10` | `networkFusion.cu:363-375` |
| HashGrid: `vectorDim ∈ {3,4}`, `featuresLevel == 2` | `TinyMLPHashGrid.cu:38-39` |
| White-bg alpha compositing (synthetic) | `DataLoader.cu:213-219` |
| Single shared intrinsic/resolution for all frames | `DataLoader.cu:94-96,125` |

---

## 3. Real-World Unbounded Scenes

### 3.1 The core gap

The README describes a **real-world video → 3D** product, but the **engine is a
bounded-box, object-centric renderer**. Everything outside the hardcoded `[-1.5,1.5]³`
AABB is clamped/invisible (`processRays.cu:291,782`). The Python "smart centering" is a
real *workaround* — it crops the world to the object and discards everything beyond the
90th-percentile radius (hence "no background fog": there is no background model). Good
for turntable objects; structurally cannot represent 360 outdoor / forward-facing
backgrounds.

### 3.2 Spatial cascades (your idea) — valid; it's the Instant-NGP recipe

Nested 128³ occupancy grids at scales 1, 2, 4, 8… over powers-of-2 AABBs, level picked
by distance from origin. This is **exactly Instant-NGP's `aabb_scale` cascade** for
unbounded scenes. Three things to get right:

- **Cascades ≠ your existing mipmaps.** `levelsMipmap=4` + `buildMipmaps()` is a
  coarse-to-fine LOD of the *same box* for empty-space skipping. Cascades are an
  **orthogonal axis**: same resolution over *nested larger boxes*. A real setup has
  both (cascade picks the box; mipmaps skip empty space within it).
- **Exponential / cone stepping is mandatory** (the part people forget). Your DDA steps
  at the finest voxel wall; marching to scale-16 that way needs thousands of samples and
  `MAX_HITS=128` truncates the ray before it reaches the background. Step size must grow
  with distance: `dt = max(dt_min, t * cone_angle)`. **Cascade + exponential stepping
  are a package deal.**
- **One hash encoding over the largest box** (not one grid per cascade — that's the
  heavier KiloNeRF/Block-NeRF design). The cascade is only for the occupancy/marching
  structure. Enlarging the encoded box costs unit-cube resolution, so bump
  `hashTableSize` (currently 2^19) and possibly `numLevels`.

### 3.3 Cascades vs scene contraction

| | **Spatial cascades (Instant-NGP)** | **Scene contraction (mip-NeRF-360)** |
|---|---|---|
| Solves | Empty-space skipping + extent over large scenes | Where representation *capacity* goes |
| Resolution falloff | Discrete power-of-2 steps | Smooth, continuous |
| Fits current code | **Yes** — reuses DDA + occupancy grid + Euclidean hash | Needs a position warp before hash lookup + non-uniform marching metric |
| 360 quality | Good | Better (SOTA) |
| Cost | Moderate | Higher |

Not mutually exclusive. Cascade = "how do I march efficiently across a huge volume."
Contraction = "how do I spend finite grid capacity." Modern best-of-both (Nerfacto,
Zip-NeRF) uses **contraction for the encoding + occupancy/proposal sampling +
exponential stepping**.

**Recommendation:** do **cascade + exponential stepping first** (lower friction, reuses
DDA + occupancy bitgrid + Euclidean hash → a *working* unbounded renderer), then add
contraction later as a quality pass.

### 3.4 Other real-world blockers (priority order)

**Tier 1 — without these, real scenes fundamentally can't work:**
- Scene contraction or NDC (positions currently hard-clamped to `[0,1]`).
- **Data-derived AABB** instead of the hardcoded box.
- A **background / sky model** (small MLP on ray direction, or learned constant).

**Tier 2 — quality/robustness on real captures:**
- Raise / make adaptive `MAX_HITS=128`; distance-adaptive stepping.
- Per-frame intrinsics + lens distortion (currently one shared `camera_angle_x`).
- **Appearance embeddings** (NeRF-W style) for auto-exposure / white-balance drift —
  a real instability source on real video.
- Ray sampling without replacement (current PCG shuffle has replacement + modulo bias).

**Tier 3 — cheap engineering hardening:**
- fp16 activation clamping in forward kernels (the source of the NaN trigger).
- Errors instead of silent no-ops on unsupported MLP config.
- `cudaGetLastError()` after kernel launches.
- Null guard on the HashGrid backward cached input pointer.

---

## 4. NeRF vs 3D Gaussian Splatting

**Is pivoting to 3DGS valid? Yes** — it's arguably the dominant direction (2023→2026)
for real-time, real-world radiance fields. But it's a **rewrite of the rendering core,
not an extension.**

**Transfers from this codebase:**
- The **COLMAP front-end** (`colmap2nerf.py` + point-cloud parsing) — and becomes *more*
  central: 3DGS **initializes Gaussians from the COLMAP sparse point cloud**.
- CUDA infra: `DeviceBuffer`, Adam (3DGS runs per-Gaussian Adam), data loading.

**Does NOT transfer:**
- The **hash grid, fused tensor-core MLP (TinyMLP — the crown jewel), DDA marcher,
  occupancy grid, volume-rendering compositing** all become irrelevant to *vanilla*
  3DGS (no MLP, no ray marching; per-Gaussian SH colors + tile-based differentiable
  rasterizer).
- New hard CUDA work: the rasterizer **backward pass** (gradients through 3D→2D
  covariance projection + per-tile alpha blending), per-tile Gaussian sorting, and
  **adaptive density control** (clone/split/prune).

**Nuance:** the field is converging on **"Gaussians + small MLPs."** **Scaffold-GS**
predicts Gaussian attributes from anchor features via a small MLP — which **would reuse
TinyMLP**. So the MLP engine isn't necessarily wasted; it moves from "the renderer" to
"an attribute predictor."

**Verdict:** 3DGS is better for **real-time rendering and fast capture**, not for clean
mesh export (see §5). Treat it as a **v2 rewrite of the renderer**, aimed at Scaffold-GS
so TinyMLP stays in play — *after* the NeRF path proves the end-to-end pipeline.

---

## 5. The End Goal: Clean Mesh Export to FBX

**Goal:** after training, export clean, editable objects to FBX that someone can open
and continue editing in Blender.

### 5.1 The key insight

"Export to FBX editable in Blender" is a pipeline, and **neither vanilla NeRF nor
vanilla 3DGS is built for it:**
- **Density NeRF → marching cubes** (Instant-NGP's mesh export): blobby, noisy,
  non-watertight, floater-ridden, million-triangle. Unusable as an editable asset.
- **3DGS → mesh**: Gaussians are fuzzy blobs, not a surface — cannot mesh directly.

The thing designed for clean meshes is **SDF-based neural surface reconstruction.**
Representing geometry as a signed distance function makes the surface the well-defined
zero-level-set → **watertight, normal-consistent, clean** meshes.

### 5.2 Is 3DGS or NeRF better for export?

**The NeRF/SDF camp wins for clean meshes — and it reuses the existing engine.**

| | Mesh quality | Reuses engine | Export maturity |
|---|---|---|---|
| Density NeRF (now) | Poor (blobby) | — | Naive baseline only |
| **SDF surface recon** (NeuS2 / Neuralangelo) | **Best — clean, watertight** | **Yes** — hash grid + MLP + marcher stay | Mature, built for this |
| 3DGS + SuGaR / 2DGS | Improving, historically noisier | No (rewrite) | Catching up |

**SDF surface reconstruction is an *evolution* of the current engine, not a rewrite.**
NeuS2 is essentially "Instant-NGP but SDF": swap the density head for an SDF head and use
NeuS-style volume rendering (surface density = logistic function of the SDF). The hash
grid, TinyMLP, DDA marcher, and occupancy grid all stay. Given the export endgame, this
is a stronger next step than either unbounded-NeRF polish or a 3DGS pivot — and it keeps
the tensor-core MLP central.

### 5.3 The export pipeline (where "clean" actually comes from — ~70% post-processing)

1. **Surface extraction** — marching cubes on the SDF zero-level-set → raw mesh.
2. **Remesh / decimate** — quadric edge-collapse or quad remeshing → clean, low-poly,
   editable topology. (This is what makes it Blender-clean.)
3. **UV unwrap + texture baking** — the hard appearance step. NeRF colors are
   **view-dependent** (SH/MLP); Blender wants **PBR materials**. Bake radiance down to a
   **diffuse albedo texture** (+ ideally normal + separated specular). Without this the
   asset looks wrong in Blender's rasterizer.
4. **FBX export** — trivial: mesh + textures → FBX/glTF/OBJ via Assimp or a Blender
   Python script. FBX is just a container.

### 5.4 "Objects" (plural) → per-object meshes

Splitting a scene into separate editable objects needs a **3D segmentation** stage
(distinct from meshing): lift 2D segmentation (e.g. SAM) into the 3D field, then mesh
each segment. Only needed if per-object splits matter rather than one scene mesh.

---

## 6. Recommended Sequencing

```
[done]  Density NeRF + NaN hardening
   │
   ├─►  (A) Unbounded NeRF: spatial cascades + exponential stepping + data-derived AABB
   │         → working real-world renderer; reuses 100% of current engine
   │
   ├─►  (B) SDF retarget (NeuS2-style): density head → SDF head + NeuS volume rendering
   │         → clean, watertight geometry; reuses hash grid + TinyMLP + marcher
   │
   ├─►  (C) Export pipeline: marching cubes → remesh/decimate → UV + bake → FBX
   │         → the actual deliverable (budget real time here; "clean" is won in remesh+bake)
   │
   └─►  (D, optional) 3DGS as a separate real-time track (target Scaffold-GS to reuse TinyMLP)
        (E, optional) SAM-based 3D segmentation for per-object export
```

For the **export endgame specifically**, the critical path is **B → C** (SDF + export
pipeline). Unbounded (A) matters if scenes have meaningful backgrounds; it can run in
parallel or be deferred for object-centric captures.

---

## 7. Reading List (Master)

### Foundations / current engine
- **Instant-NGP** — Müller et al., SIGGRAPH 2022. *Multiresolution hash encoding;
  occupancy-grid cascade + exponential stepping for unbounded (your cascade blueprint).*
- **Mip-NeRF** — Barron et al., ICCV 2021. *Cone tracing / anti-aliasing foundation.*

### Real-world / unbounded
- **Mip-NeRF 360** — Barron et al., CVPR 2022. *Scene contraction, proposal MLP,
  distortion regularizer (kills floaters).*
- **NerfAcc** — Li et al., ICCV 2023. *Cleanest modern description of occupancy-grid +
  proposal sampling and exponential stepping; closest to your code.*
- **Nerfstudio / Nerfacto** — Tancik et al., SIGGRAPH 2023. *The engineering recipe:
  contraction + proposal sampler + hash grid + appearance embeddings + pose refinement.*
- **Zip-NeRF** — Barron et al., ICCV 2023. *Anti-aliased hash grids + contraction; NeRF
  SOTA on unbounded.*
- **NeRF-W** — Martin-Brualla et al., CVPR 2021. *Per-image appearance embeddings for
  varying exposure/lighting (real video auto-exposure).*
- **BARF** — Lin et al., ICCV 2021. *Joint pose refinement for noisy COLMAP poses.*

### Clean surface / mesh export (SDF)
- **NeuS** — Wang et al., NeurIPS 2021. *SDF-as-volume-rendering; the conceptual base.*
- **NeuS2** — Wang et al., ICCV 2023. *NeuS at Instant-NGP speed with hash grids —
  your direct retarget target.*
- **Neuralangelo** — Li et al., CVPR 2023. *Hash-grid SDF + numerical gradients +
  coarse-to-fine; highest-fidelity surfaces.*
- **VolSDF** — Yariv et al., NeurIPS 2021. *Alternative SDF↔density formulation.*

### Appearance baking (materials half of export)
- **BakedSDF** — Yariv et al., SIGGRAPH 2023. *SDF → mesh + view-dependent appearance
  for real-time/editable assets; most on-topic for export.*
- **NeRF2Mesh** — Tang et al., ICCV 2023. *Explicit textured-mesh extraction +
  refinement.*
- **MobileNeRF** — Chen et al., CVPR 2023. *NeRF as textured polygons; bake-to-texture
  mindset.*

### 3DGS (optional v2 track)
- **3D Gaussian Splatting** — Kerbl et al., SIGGRAPH 2023. *The original.*
- **2D Gaussian Splatting** — Huang et al., SIGGRAPH 2024. *Surfels; better geometry /
  cleaner mesh extraction.*
- **SuGaR** — Guédon & Lepetit, CVPR 2024. *Surface-aligned Gaussians → Poisson mesh;
  demos Blender editing.*
- **Scaffold-GS** — Lu et al., CVPR 2024. *Neural Gaussians via a small MLP — reuses
  TinyMLP.*

### Per-object segmentation (optional)
- **Panoptic Lifting** — Siddiqui et al., CVPR 2023. *2D panoptic → consistent 3D field.*
- **SA3D** — Cen et al., NeurIPS 2023. *Segment Anything (SAM) in 3D for NeRF.*
- **Gaussian Grouping** — Ye et al., ECCV 2024. *3DGS-side analog.*

### Tooling references
- **Marching Cubes** — Lorensen & Cline, 1987.
- **Instant Meshes** — Jakob et al., SIGGRAPH Asia 2015. *Quad remeshing.*
- **gsplat** — reference 3DGS rasterizer implementation.
</content>
</invoke>
