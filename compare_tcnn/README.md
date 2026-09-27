# TinyMLP vs tiny-cuda-nn

Head-to-head benchmark of this repo's `modules/TinyMLP` against NVIDIA's
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) (BSD-3-Clause), built into one executable
and timed with identical code. Full numbers: [`results/report.md`](results/report.md).

## Run it

```powershell
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn compare_tcnn/tiny-cuda-nn   # gitignored
cmake -S compare_tcnn -B compare_tcnn/build -DCMAKE_CUDA_ARCHITECTURES=89 -DTCNN_CUDA_ARCHITECTURES=89
cmake --build compare_tcnn/build --config Release -j 16          # ~15 min, tcnn dominates
cd compare_tcnn
./build/Release/bench_tcnn.exe --speed results/speed.csv --conv results/conv.csv --ablate results/ablate.csv   # ~20 min
python make_report.py --meta "GPU / driver / commit notes"
```

`--quick` runs one config per model plus a 200-step convergence check (about 1 minute).

## What was compared

- **Frameworks:** TinyMLP; tcnn precompiled `FullyFusedMLP` + `HashGrid`; tcnn 2.x JIT
  (`set_jit_fusion(true)`, which fuses the encoding, MLP and the whole training step into one
  runtime-compiled kernel).
- **Models:**
  - plain MLP, with hidden 32/64/128 and 2/4 layers
  - hash grid (16 levels, 2^19 table) + MLP with 16 outputs
  - 3DMaker's colour head (32→32→32→3, sigmoid)
- **Batches:** 64K, 256K and 1M.
- **Ops:** inference; backward; full train step (forward + L2 loss + backward + Adam).
- **Convergence:** the same hash-grid model, data stream and Adam settings, fitting a 16-channel
  multi-frequency field; test MSE is tracked per step and per second.
- **Ablation (`--ablate`):** TinyMLP with biases frozen, then also with tcnn's initialisation, on
  offset and zero-mean fields, to explain the quality difference.

## Findings (RTX 4060 Laptop, CUDA 13.2, tcnn 0109538, TinyMLP 6212224)

**TinyMLP is in tiny-cuda-nn's class.** It is faster on some operations and slower on others,
and it converges to a lower error.

| | vs tcnn precompiled | vs tcnn JIT |
|---|---|---|
| Plain MLP inference | **1.69× faster** | 0.71× (JIT faster) |
| Plain MLP backward | **1.37× faster** | n/a |
| Plain MLP train step | **1.08× faster** | 0.98× (parity) |
| Hash grid inference | 0.99× (parity; **1.11× faster** at 3DMaker's config) | 0.86× |
| Hash grid backward | 0.63× (tcnn 1.6× faster) | n/a |
| Hash grid train step | 0.75× | **1.48× faster** |

Geometric means over all hidden/layer/batch configs; >1 means TinyMLP is faster.

TinyMLP is driven its intended fast way:
- Hash-grid positions are passed as 4 floats (x, y, z, pad), as 3DMaker does, so no pad kernel runs.
- The train step calls `forward(in, nullptr, ...)`, so no outputs are copied back to the caller.
  tcnn's `training_step` returns nothing to the caller either.

The first run fed 3-float positions and copied outputs out; it is kept as `results/speed_v1.csv`.
Those two changes alone made TinyMLP hash-grid inference ~7% faster and the plain-MLP train step
~18% faster.

**Quality: equal, and TinyMLP's apparent edge came from its biases.**

- On the first test field (0.5 + 0.4·sin), TinyMLP ended with lower error at 3000 steps:
  0.00990 test MSE vs 0.01073 for tcnn (about 8% lower).
- A follow-up ablation (`--ablate`, table at the end of `results/report.md`) traced that gap to
  the biases. With biases frozen at 0, TinyMLP ends at 0.01042; adding tcnn's initialisation as
  well gives 0.01062. Both are on par with tcnn's 0.01052 in that run.
  - Output biases learn the constant +0.5 offset directly, while tcnn's bias-free MLP has to
    build it out of ReLU features.
- On a zero-mean field (0.4·sin), biases make no difference: TinyMLP 0.00999, without biases
  0.00996, tcnn 0.00988 (within noise).
- The two libraries' training updates are equally good. tcnn's fp16 hash gradients cost it
  nothing measurable here, which is evidence that TinyMLP could switch to fp16 hash-gradient
  atomics (gap #1 below) without losing quality.
- Initialisation only affects the first few hundred steps:
  - tcnn's ±1e-4 hash-table start is faster early on the offset field (step 100: 0.033 vs
    0.050, both without biases).
  - Both converge to the same place.
- For NeRF, keep the biases: colour and density outputs have per-channel offsets, and biases cost
  almost nothing. Just don't claim TinyMLP "learns better" in general.

**Where the speed gaps come from.** These are diagnoses from the data and source, not yet
confirmed by experiments.

1. **Hash-grid backward (the main gap):**
   - tcnn accumulates hash gradients as fp16 with one `atomicAdd(__half2)` per corner
     (`tiny-cuda-nn/include/tiny-cuda-nn/encodings/grid.h:665`), which its loss scale of 128
     makes safe.
   - TinyMLP issues two fp32 `red.global.add` per corner (`networkFusionBackwardHashtable.cu:883-884`, the default split-scatter path).
     That is twice the atomics and a gradient buffer twice the size.
   - fp16 atomics were rejected earlier only because a loss scale of 65536 overflows fp16.
     Lowering the hash-table loss scale to about 128 and switching to half2 reductions is
     the experiment to run.
2. **Copies in `TinyMLP::forward`:**
   - With `d_outputs = nullptr`, the output copy (`TinyMLP.cu:276`) is skipped. That alone
     took the plain-MLP train step from 10.1 to 7.8 ms at 1M/64/2, now 1.08× faster than tcnn.
   - 3DMaker does need the outputs, so the copy should go at the source: have the kernel write
     straight into the caller's buffer.
   - The input snapshot (`TinyMLP.cu:271`) still runs on every forward. The API gives no way
     to skip it; it needs an "input stays untouched until backward" contract.
3. **Small-output heads:**
   - The colour head writes 8-wide padded fp32 and then runs a separate unpad kernel
     (`TinyMLP.cu:314`). That is why its inference is 0.31× tcnn at 256K.
   - tcnn JIT fuses the colour head's whole train step into one kernel: 0.18 ms vs 0.85 ms.
   - Caveat: at 256K rows the colour head's ~17 MB input fits in the 4060's 32 MB L2 cache,
     so all numbers at that size are flattered by repeated launches.
4. **tcnn JIT is not uniformly better.**
   - It wins on small MLPs and inference.
   - It is 1.5–2× slower than both TinyMLP and precompiled tcnn at training hash grids,
     where the fused kernel's gradient scatter dominates.

**Parity caveats:**
- tcnn's MLP has no biases.
- Each library uses its own default loss scale.
- Speed runs give TinyMLP 4-float positions. The convergence and ablation runs use 3-float
  positions, which only affects speed, not quality.
- The results come from one laptop GPU, which can throttle thermally. Configs were
  interleaved and the median of 3 runs is reported, but expect ±5% noise.

**Fair pitch line:** "Our from-scratch TinyMLP matches NVIDIA's tiny-cuda-nn: 1.4–1.7× faster
on plain-MLP inference and backward, on par for hash-grid inference, ~25% slower on
hash-grid training, and it reaches the same quality on the same data."
