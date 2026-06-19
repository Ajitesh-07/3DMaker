# 3DMaker

**3DMaker** is an end-to-end Neural Radiance Field (NeRF) engine written **from scratch in C++ and
CUDA** — no PyTorch, no TensorFlow, no tiny-cuda-nn. Point it at an `.mp4` or a folder of photos and
it runs the full pipeline — structure-from-motion, neural training, and real-time rendering — to turn
a casual capture into an explorable 3D scene.

Architecturally it is an **Instant-NGP-class** system: a custom multi-resolution hash grid, a pair of
fully-fused **tensor-core MLPs**, and a **hit-centric DDA raymarcher** over a hierarchical occupancy
grid. The neural core (`TinyMLP`) is framework-free and fast enough to train scenes on a single
consumer GPU in seconds.

> **Scope (honest):** today this is a **bounded-box, object-centric** renderer — superb for turntable
> objects and front-facing captures. Full unbounded/360-outdoor support and clean mesh→FBX export are
> the active roadmap (`ROADMAP.md`, `docs/realworld_quality_roadmap.md`).

> **Headline:** an indoor scene trains in **under 4 minutes** using **under 2 GB of VRAM** on a
> **laptop RTX 4060** — with zero deep-learning frameworks, the whole stack hand-written in CUDA/C++.

---

## 📑 Table of Contents

- [Pipeline at a glance](#-pipeline-at-a-glance)
- [Results](#-results)
- [Quick start](#-quick-start)
- [Repository layout](#-repository-layout)
- [Technical achievements](#-technical-achievements)
  - [TinyMLP — the math engine](#1-tinymlp--the-math-engine)
  - [NeRF engine](#2-nerf-engine)
  - [Preprocessing & tooling](#3-preprocessing--tooling)
  - [Apps](#4-apps)
- [Requirements](#-requirements)
- [Building](#-building)
- [Tests & benchmarks](#-tests--benchmarks)
- [Status & roadmap](#-status--roadmap)

---

## 🔭 Pipeline at a glance

```
 video.mp4 / images/        scripts/colmap2nerf.py            modules/NeRF (INerfTrainer)
 ──────────────────►  FFmpeg ─► COLMAP SfM ─► transforms  ─►  hash grid + occupancy grid
                      (frames)  (poses+points)  _train.json     + tensor-core MLPs (train)
                                                                          │
                                          ┌───────────────────────────────┤
                                          ▼                               ▼
                                   model.inerf  ──►  3DViewer.exe   nerf_360.mp4 + frames/
                                  (weights + scene)   (interactive)   (showcase render)
```

Every stage is driven by **`3DMaker.exe`** — you only supply the capture.

---

## 📊 Results

The point of building the engine from scratch isn't just that it works — it's that it's **lean enough
to run on a laptop**. Reference numbers on a **consumer mobile RTX 4060 (8 GB, 115 W laptop GPU)**:

| Metric | Result |
|---|---|
| **Train time** | **< 4 minutes** for an indoor scene (full convergence) |
| **Peak VRAM** | **< 2 GB** during training |
| **Per-step latency** | ~10 ms/step baseline (falls as the occupancy grid prunes empty space) |
| **Hash-grid throughput** | 100M+ multi-resolution evaluations/sec |
| **Dependencies** | **none** — no PyTorch / TensorFlow / tiny-cuda-nn; pure CUDA + C++ |

**Why those numbers are notable:**

- **From scratch.** Every kernel — the fused tensor-core MLP, the hash-grid encode/backward, fused
  Adam, the DDA raymarcher — is hand-written. There is **no deep-learning framework anywhere** in the
  training or rendering path.
- **Fits a laptop.** The **hit-centric compaction** pipeline (only active samples are materialized)
  plus FP16 tensor-core math keeps peak VRAM **under 2 GB**, so it trains on an 8 GB mobile GPU with
  room to spare — where framework-based NeRFs typically ask for 6–24 GB.
- **Minutes, not hours.** Fully-fused single-launch MLPs, occupancy-grid empty-space skipping, and
  Morton-ordered cache-coherent lookups bring a real indoor capture to convergence in **single-digit
  minutes** on hardware most people already own.

> Times scale with capture size, resolution, and `--steps`; the figures above are a representative
> indoor scene at default settings. Reproduce with `3DMaker.exe train --data <scene> --validate`.

---

## 🚀 Quick start

### 1. Build

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

This produces two executables in `build/Release/`:

| Executable | Source | Role |
|---|---|---|
| `3DMaker.exe`  | `src/main.cpp`  | CLI orchestrator: COLMAP → train → save → (optional) showcase video |
| `3DViewer.exe` | `src/render.cpp`| Real-time interactive viewer |

> The default build targets **sm_89 (Ada / RTX 40-series)**. For another GPU, edit
> `CMAKE_CUDA_ARCHITECTURES` in the root `CMakeLists.txt` (e.g. `120` for Blackwell / RTX 50-series,
> `86` for Ampere).

### 2. Train

Pass a **video file** or a **directory containing an `images/` subfolder**. If a
`transforms_train.json` doesn't already exist, 3DMaker runs the COLMAP pipeline automatically first.

```powershell
.\build\Release\3DMaker.exe train --data data\my_capture.mp4 --validate --video
```

**Options** (all optional):

| Flag | Meaning | Default |
|---|---|---|
| `--steps N`     | training-step cap | `50000` |
| `--epochs N`    | outer-loop epoch cap (step-bounded) | `100` |
| `--cascades N`  | occupancy cascades; `0` = estimate from dataset | `0` |
| `--lambda F`    | distortion-loss weight (floater suppression) | `0.1` |
| `--K N`         | sub-voxel samples per occupied voxel (anti-aliasing) | `1` |
| `--max-lr F`    | cosine LR schedule start | `0.01` |
| `--min-lr F`    | cosine LR schedule floor | `0.0001` |
| `--validate`    | hold out every 8th image, report test PSNR | off |
| `--video`       | render the 360° orbit `.mp4` after training | off |

Outputs are written **next to the dataset**:

```
data/my_capture/
├── transforms_train.json     # camera poses from COLMAP
├── model.inerf               # trained weights + scene footer (center/up/scale/fov)
├── frames/                   # rendered train-view PNGs (+ orbit frames if --video)
└── nerf_360.mp4              # showcase orbit (only with --video)
```

During training you get a live progress bar:

```
[##########          ] 24500/50000 steps | epoch 1.82 | 9.4 ms/step | ETA 04:01 | PSNR 27.3
```

### 3. View

```powershell
.\build\Release\3DViewer.exe data\my_capture\transforms_train.json data\my_capture\model.inerf
```

Blender-style controls: **left-drag** to orbit, **middle-drag** to pan, **scroll** to zoom. The viewer
reads the camera FOV from `transforms_train.json` and restores the scene framing from the `.inerf`
footer, so it opens looking at the subject.

---

## 🗂 Repository layout

```
src/
  main.cpp              3DMaker.exe — training CLI / orchestrator (drives the INerfTrainer facade)
  render.cpp            3DViewer.exe — real-time OpenGL + CUDA-interop viewer

modules/TinyMLP/        Framework-free math engine (knows nothing about NeRF):
  TinyMLP.{h,cu}            fused tensor-core MLP (color head)
  TinyMLPHashGrid.{h,cu}    multi-resolution hash grid + fused MLP (density head)
  networkFusion*.cu         fused WMMA forward / backward kernels
  optimizerKernel.cu        fused Adam (weights, biases, hash grid)

modules/NeRF/           Scene representation, rendering & training:
  NerfTrainer.{h,cu}        INerfTrainer — the public facade (init/train/validate/save/load/render)
  InstantNerf.cu            occupancy grid, hit-centric train/render loops, .inerf I/O
  processRays.cu            DDA raymarcher + hit compaction over the occupancy bitgrid
  compositing.cu            volume rendering + distortion loss
  DataLoader.cu             COLMAP transforms parsing + pinned-memory ray streaming
  RenderKernels.cu          camera ray generation, tonemapping

scripts/
  colmap2nerf.py         video/images → COLMAP SfM → transforms_train.json (smart object centering)
  analyze_capture.py     capture-quality gate (camera-path coverage)

third_party/colmap/     bundled COLMAP (Windows); 'colmap' on PATH is the fallback
docs/, ROADMAP.md       design notes & forward plan
```

The split is deliberate: **`TinyMLP` is a generic, reusable fused-MLP + hash-grid library** with its
own benchmarks; **`NeRFModule` orchestrates it** into a renderer.

---

## 🏆 Technical achievements

### 1. `TinyMLP` — the math engine

A from-scratch, framework-free implementation of fused tensor-core MLPs and a multi-resolution hash
grid — the part that replaces PyTorch/tiny-cuda-nn entirely.

- **Fully-fused tensor-core MLP.** The whole network runs in **one kernel launch**: hidden
  activations stay resident in **shared memory** across every layer (`networkFusion.cu`), so there is
  no round-trip to global memory between layers. Matrix multiplies use hardware **Tensor Cores via
  `nvcuda::wmma`** in native FP16 with **FP32 accumulators**.
- **Asynchronous weight prefetch.** Uses `cp.async` (LDGSTS) 128-bit loads with **2-deep pipelining**
  to stream the next layer's weights from global → shared memory while the math units work the current
  tile, hiding memory latency behind compute.
- **Mixed precision done right.** Keeps **FP32 master weights** for stable optimizer updates plus
  **FP16 forward copies** for tensor-core throughput — the standard mixed-precision recipe, hand-rolled.
- **Fused Adam optimizer.** Weight, bias **and hash-grid** Adam updates are fused, with bias
  correction and loss-scaling, plus **NaN/Inf guards** that prevent a single bad gradient from latching
  the moments permanently (`optimizerKernel.cu`).
- **Multi-resolution hash grid (`TinyMLPHashGrid`).** Hash encoding + MLP fused into a single
  forward/backward, with a dense-vs-hashed level split and Adam directly over the FP32 master table.
  Configurable table size, level count, growth factor `b`, base resolution, and features-per-level.
- **Micro-optimized kernels.** `PAD=8` shared-memory layout to kill bank conflicts; `uint32_t`
  indexing so `nvcc` synthesizes bitwise `SHR`/`AND` instead of integer div/mod; `__launch_bounds__`
  for occupancy; warp topology folded to match the hidden dimension for full SM utilization.
- **Training / inference modes.** Inference path skips activation storage to cut VRAM; weights
  load/save for checkpointing.
- **Reported throughput:** **100M+ multi-resolution hash-grid evaluations/sec** on a consumer GPU.

*Configurable envelope:* hidden dim ∈ `{16, 32, 64, 128}`, up to 10 layers (auto-padded to powers of
two); hash grid `featuresPerLevel = 2`, input `vectorDim ∈ {3, 4}`; output activation none/sigmoid.

### 2. NeRF engine

The scene representation, raymarcher, and training/rendering loops built on top of `TinyMLP`.

- **Hit-centric compaction pipeline.** Instead of allocating worst-case samples per ray, the engine
  **dynamically packs only the active (occupied) samples** in memory before evaluating the MLPs —
  dramatically reducing VRAM versus naïve per-ray allocators and keeping the tensor cores saturated.
- **Hierarchical occupancy grid + DDA marcher.** A `128³` occupancy **bitgrid** with **mipmap
  max-pooling** (4 levels) lets a custom Digital Differential Analyzer raymarcher **skip empty space**
  coarse-to-fine and jump straight to surfaces. The grid is maintained online with **EMA density
  updates** and early/late update schedules.
- **Spatial cascades.** Nested occupancy grids over powers-of-two AABBs (the Instant-NGP
  `aabb_scale` recipe) to extend reach beyond the unit cube; cascade count auto-estimated from scene
  radius.
- **Morton-ordered samples.** Active samples are Z-order sorted so hash-grid lookups hit coherent
  cache lines.
- **Two MLP heads.** A hash-grid **density** network (`TinyMLPHashGrid`) and a **color** network
  (`TinyMLP`) fed degree-3 **spherical-harmonic** directional encoding for view-dependent shading.
- **Volume rendering with distortion regularization.** Front-to-back alpha compositing plus a
  **mip-NeRF-360-style distortion loss** (`--lambda`) that collapses floaters and "fog."
- **Anti-aliasing knob.** `--K` sub-voxel samples per occupied voxel (a Zip-NeRF-style multisample
  primitive).
- **Anneal-and-hold LR schedule.** Cosine decay `max-lr → min-lr` over training, then held at the
  floor — restoring intended late-training stability.
- **Held-out validation.** `--validate` reserves every 8th image and reports true test-set PSNR.
- **Self-contained checkpoints (`.inerf`).** Saves all weights **plus a scene footer** (center, up,
  scale, camera FOV) so the viewer reproduces the exact framing after a bare load — no JSON needed at
  view time.
- **Clean public facade (`INerfTrainer`).** `init / loadDataset / train / validate / save / load /
  renderOrbit / renderTrainView`, with a live progress bar (steps, epoch %, **EMA ms/step**, ETA,
  PSNR). Both `3DMaker.exe` and the dev harness `train_hit.exe` are thin wrappers over it.
- **TRAINING / INFERENCE memory modes** trade activation storage for VRAM at render time.

### 3. Preprocessing & tooling

- **`colmap2nerf.py` — one-command SfM.** Extracts frames (FFmpeg), runs the full COLMAP pipeline
  (feature extraction → exhaustive matching → SfM mapping) with a **single shared intrinsic**
  (`single_camera=1`, correct for handheld phone video), and emits `transforms_train.json`.
- **Smart object centering.** Rather than centering on the camera centroid (which breaks
  front-facing captures), it parses the COLMAP **`points3D`** cloud, filters background by view-track
  length, and centers the neural bounding box on the **physical subject** via least-squares ray
  convergence — then rotates *up* → `+Z` and scales the scene to the unit cube. Result: maximum
  resolution on the object and no background fog.
- **`analyze_capture.py` — capture-quality gate.** Scores a capture's camera-path coverage (azimuth
  arc, latitude crossing the equator) to flag captures that physically can't reconstruct well *before*
  you spend GPU time.

### 4. Apps

- **`3DMaker.exe` (orchestrator).** Ingests a video/folder, runs COLMAP if needed, trains, renders
  sample train views, optionally renders a 360° orbit and encodes it to `.mp4` (FFmpeg), and saves the
  `.inerf` model — all from one command.
- **`3DViewer.exe` (visualizer).** A native **GLFW + OpenGL** viewer with **zero-copy CUDA–GL
  interop**: the network renders pixels straight into the GPU display buffer (PBO) — no CPU round-trip
  — for interactive orbiting/panning. Reconstructs the FOV and opening pose from the dataset and the
  `.inerf` footer.

---

## 📦 Requirements

| Dependency | Notes |
|---|---|
| **NVIDIA GPU** | Volta+ (SM 7.0+) for Tensor Cores; default build targets Ada (sm_89) |
| **CUDA Toolkit** | built/tested with **13.2**; 12.x should work (CCCL needs the conforming MSVC preprocessor — handled in CMake) |
| **C++20 compiler** | MSVC v143 / VS 2022+ (primary), or `g++-11`+ on Linux |
| **CMake** | ≥ 3.24 |
| **Python 3** + numpy | for `colmap2nerf.py` / `analyze_capture.py` |
| **COLMAP** | bundled under `third_party/colmap/` on Windows; install separately and put on PATH for Linux |
| **FFmpeg** | must be on PATH (video frame extraction + showcase `.mp4` encode) |
| GLFW / FreeType / RmlUi | fetched automatically by CMake (`FetchContent`) — no manual install |

---

## 🛠 Building

### Windows (primary)

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
# → build\Release\3DMaker.exe and build\Release\3DViewer.exe
```

### Linux

```bash
git clone <repo-url> 3DMaker && cd 3DMaker
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

> Install COLMAP and FFmpeg via your package manager (`apt install colmap ffmpeg`) and ensure they're
> on PATH. Adjust `CMAKE_CUDA_ARCHITECTURES` in the root `CMakeLists.txt` for your GPU.

---

## 🧪 Tests & benchmarks

Both modules ship standalone test/benchmark targets, gated behind a CMake option.

**TinyMLP** (hash-grid & matmul throughput, forward/backward correctness):

```bash
cmake -S modules/TinyMLP -B modules/TinyMLP/build -DCMAKE_BUILD_TYPE=Release -DTINYMLP_BUILD_TESTS=ON
cmake --build modules/TinyMLP/build --config Release -j
# e.g. test_forward, test_backward, test_mlp, test_optimizer, test_modes, demo
```

**NeRF** (training harness + round-trip tests):

```bash
cmake -S modules/NeRF -B modules/NeRF/build -DCMAKE_BUILD_TYPE=Release -DNERF_BUILD_TESTS=ON
cmake --build modules/NeRF/build --config Release -j
# train_hit  — direct dataset training harness:  train_hit.exe <dataset_path> [lambdaDist]
# test_scene_footer — .inerf save/load round-trip
```

---

## 🧭 Status & roadmap

3DMaker today is a strong **object-centric** NeRF: hash grid, occupancy grid, distortion loss, and a
COLMAP front-end with smart subject centering. The forward plan is documented in:

- **`ROADMAP.md`** — code review, unbounded-scene plan (cascades + exponential stepping + scene
  contraction), the 3DGS question, and the **clean mesh → FBX** end goal (SDF surface reconstruction).
- **`docs/realworld_quality_roadmap.md`** — quality/efficiency techniques for real-world captures
  (depth supervision, per-image appearance, Zip-NeRF anti-aliasing, pose refinement), framed as "we've
  independently built ~half of nerfacto."

Inspired by **Instant-NGP** (Müller et al., 2022) and the broader mip-NeRF-360 / Nerfstudio line.
```