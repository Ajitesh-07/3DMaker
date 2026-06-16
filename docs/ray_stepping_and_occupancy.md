# Exponential Ray Stepping & Optical-Thickness Occupancy

*Instant-NGP appendix E.1 (step size) and E.2 (occupancy grids), reconciled with the paper, the
reference source, and our own DDA marcher + cascaded occupancy grid.*

This note answers three questions that came up while debugging real-world scenes:

1. **Where do you place samples along a ray?** (E.1 — exponential / cone stepping)
2. **How do you skip the empty space between them?** (E.2 — the occupancy grid)
3. **What is the `0.01` threshold, why is it multiplied by a per-cascade `Δt`, and why
   `min(0.01, mean)`?** (the optical-thickness threshold + mean bootstrap)

Everything here is cross-checked against the NVlabs source (`nerf_device.cuh`,
`testbed_nerf.cu`) so the constants are exact, not remembered.

---

## 0. The problem both tricks solve

A volume-rendered ray needs samples $t_0 < t_1 < \dots < t_{N-1}$ at which to query the
density/color field, then composites

$$ C = \sum_i T_i\,\alpha_i\,c_i,\qquad \alpha_i = 1-e^{-\sigma_i\,\delta_i},\qquad
T_i=\prod_{j<i}(1-\alpha_j),\qquad \delta_i = t_{i+1}-t_i. $$

Two cost drivers:

- **Sample spacing $\delta_i$.** Uniform spacing fine enough for nearby surfaces wastes
  enormous numbers of samples on far-away background. A pixel is a *cone*, not a ray: its
  world footprint grows linearly with distance $t$, so the *useful* sample spacing should grow
  with $t$ too. → **E.1, exponential stepping.**
- **Empty space.** Most samples land in vacuum ($\sigma\approx 0$) and contribute nothing.
  Knowing in advance which cells are empty lets you skip them entirely. → **E.2, occupancy
  grid.**

---

## 1. E.1 — Step size: cone tracing → exponential stepping

### 1.1 The one-line rule

Instant-NGP's step size is, conceptually,

$$ \boxed{\;\delta(t) \;=\; \operatorname{clamp}\big(c\cdot t,\; \delta_{\min},\; \delta_{\max}\big)\;} $$

where $c$ = `cone_angle_constant`. That is: **step proportionally to distance**, with a floor
and a ceiling. Three regimes:

| region | condition | step | meaning |
|---|---|---|---|
| near | $c\,t < \delta_{\min}$ | $\delta_{\min}$ (constant) | fine uniform stepping close to camera |
| mid  | $\delta_{\min}\le c\,t\le \delta_{\max}$ | $c\,t$ (grows ∝ $t$) | **exponential** stepping |
| far  | $c\,t > \delta_{\max}$ | $\delta_{\max}$ (constant) | clamp to coarsest cell width |

**Why "exponential":** in the middle region each step advances $t$ by a fixed *fraction* $c$:

$$ t_{n+1} = t_n + c\,t_n = (1+c)\,t_n \quad\Longrightarrow\quad t_n = t_0\,(1+c)^n . $$

Sample positions form a geometric series — uniformly spaced on a *log* axis. The number of
samples to cross a distance range $[t_a, t_b]$ is $\log_{1+c}(t_b/t_a)$, i.e. **logarithmic in
range**, not linear. That is what makes unbounded 360° scenes tractable.

### 1.2 The exact source form (`nerf_device.cuh`)

The reference code doesn't write `clamp(c·t, …)` directly. It defines an invertible warp
("stepping space") so it can (a) advance an integer number of steps in O(1) and (b) DDA across
grid cells. The two are *mathematically identical* in the middle region; the warp just makes
near/mid/far one smooth, invertible function.

```cpp
inline NGP_HOST_DEVICE float calc_dt(float t, float cone_angle) {
    return advance_n_steps(t, cone_angle, 1.0f) - t;
}
inline NGP_HOST_DEVICE float advance_n_steps(float t, float cone_angle, float n) {
    return from_stepping_space(to_stepping_space(t, cone_angle) + n, cone_angle);
}

inline NGP_HOST_DEVICE float to_stepping_space(float t, float cone_angle) {
    if (cone_angle <= 1e-5f) return t / MIN_CONE_STEPSIZE();          // bounded: linear
    float log1p_c = logf(1.0f + cone_angle);
    float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
    float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
    float at = expf(a * log1p_c);   // = MIN_CONE_STEPSIZE / ln(1+c)  ≈ δ_min / c
    float bt = expf(b * log1p_c);   // = MAX_CONE_STEPSIZE / ln(1+c)  ≈ δ_max / c
    if      (t <= at) return (t - at) / MIN_CONE_STEPSIZE() + a;      // near: linear
    else if (t <= bt) return logf(t) / log1p_c;                      // mid:  logarithmic
    else              return (t - bt) / MAX_CONE_STEPSIZE() + b;      // far:  linear
}
// from_stepping_space is the exact inverse (the same three pieces, swapped).
```

**Reading it.** In the middle region $n = \log_{1+c}(t)$, so

$$ \texttt{advance\_n\_steps}(t,c,1)=(1+c)^{\,n+1}=(1+c)\,t,\qquad
   \texttt{calc\_dt}=(1+c)t-t=c\,t, $$

recovering $\delta = c\,t$ exactly. The crossover $a_t = \delta_{\min}/\ln(1+c)\approx
\delta_{\min}/c$ is precisely the distance at which $c\,t$ would fall below the floor
$\delta_{\min}$; below it the warp is linear with slope $\delta_{\min}$ (constant step). Same
story at $b_t\approx\delta_{\max}/c$ for the ceiling. So the fancy warp **is** the boxed
clamp rule — just written so a single integer `n` indexes every sample and stepping is O(1)
invertible.

### 1.3 The constants (exact)

```cpp
NERF_STEPS()            = 1024                       // "finest steps per unit length"
SQRT3()                 = 1.73205080757
STEPSIZE()              = SQRT3()/NERF_STEPS()        ≈ 1.6914e-3   // = MIN_CONE_STEPSIZE
NERF_GRIDSIZE()         = 128                         // occupancy grid is 128^3
NERF_CASCADES()         = 8                           // max; K∈[1,5] actually used
MAX_CONE_STEPSIZE()     = STEPSIZE()·2^(C-1)·1024/128 = √3·2^(C-1)/128
                        = √3 ≈ 1.732  (for C=8)        // width of the coarsest cell
NERF_MIN_OPTICAL_THICKNESS() = 0.01
cone_angle_constant     ≈ 1/256 ≈ 0.00390625          // 0 for bounded unit-cube scenes
```

- $\delta_{\min}=\sqrt3/1024$: the grid is $128^3$ over the unit cube, and $\sqrt3$ is the
  cube's diagonal, so $\delta_{\min}$ is $1/1024$ of the diagonal ≈ **8 samples per grid cell**
  along the diagonal. Fine.
- $\delta_{\max}=\sqrt3$ (full diagonal): one step can cross the whole coarsest cascade.
- **`cone_angle_constant = 0` ⟹ `to_stepping_space` takes the first branch ⟹ pure uniform
  stepping** ($\delta=\delta_{\min}$ everywhere). That is exactly the "original NeRF, unit-cube"
  mode. Synthetic scenes use 0; real/large scenes use ~1/256. One flag toggles the whole
  behavior.
- `calc_cone_angle(...)` in current source just returns the constant — the cone half-angle is
  *not* presently derived from focal length / pixel size, even though the motivation (a pixel's
  cone) is geometric. It's a tunable scalar.

---

## 2. E.2 — Occupancy grid: skipping the empty space

Paper text (appendix E.2), verbatim in substance:

> A cascade of $K$ multiscale occupancy grids, $K=1$ for synthetic scenes, $K\in[1,5]$ for
> larger real scenes. Each grid is $128^3$, spanning a geometrically growing domain
> $[-2^{k-1}+0.5,\,2^{k-1}+0.5]^3$ centred at $(0.5,0.5,0.5)$. Each cell stores one occupancy
> bit; cells are in **Morton (z-curve) order** for memory-coherent **DDA** traversal. A sample
> placed by the step rule above is **skipped if its cell bit is low**. Grids are updated every
> 16 iterations: decay every cell's density by **0.95**, set $M$ random candidate cells to
> $\max(\text{current}, \sigma_{\text{model}})$, then re-threshold the bits.

So the occupancy grid is a coarse, conservative cache of "is there anything here?" updated
slowly (EMA decay 0.95) and sampled stochastically so it tracks the moving field without being
re-evaluated densely every step.

### 2.1 The threshold — this is the part that bit us

The bitfield is built (in `grid_to_bitfield`) as:

```cpp
float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);   // min(0.01, mean)
bits |= grid[i] > thresh ? bit : 0;
```

and the *value stored in the grid* is **optical thickness**, not raw density
(`splat_grid_samples_nerf_max_nearest_neighbor`):

```cpp
float mlp = network_to_density(network_output[i], density_activation);     // σ
float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);       // σ · δ_min · 2^level
atomicMax(&grid_out[idx], optical_thickness);                              // level = idx / 128^3
```

`scalbnf(x, level) = x · 2^level`, and `level` is the **cascade index**. So the grid stores

$$ g \;=\; \sigma \cdot \underbrace{\delta_{\min}\,2^{\text{level}}}_{\Delta t_{\text{level}}}
        \;=\; \sigma\cdot\Delta t_{\text{level}} . $$

That is **the optical thickness of one step through that cell**: $\sigma\,\Delta t$. And since
$\alpha = 1-e^{-\sigma\Delta t}\approx \sigma\Delta t$ for small values,

$$ \text{keep cell} \iff \sigma\,\Delta t_{\text{level}} > 0.01
   \iff \alpha_{\text{per step}} \gtrsim 0.01 . $$

**`0.01` is an opacity, not a density.** "Any alpha below this is invisible and is culled."
The crucial consequence is the **per-cascade scaling**: rearranging,

$$ \text{keep} \iff \sigma > \frac{0.01}{\Delta t_{\text{level}}}
   = \frac{0.01}{\delta_{\min}}\cdot 2^{-\text{level}}
   = \underbrace{\frac{0.01\cdot 1024}{\sqrt3}}_{\approx\,5.916}\cdot 2^{-\text{level}}. $$

So the **density** threshold is $\approx 5.92$ at the base cascade (this is the paper's
"$t = 0.01\cdot 1024/\sqrt3$") and is **halved for every coarser cascade**. Coarser cascades
accept lower density because a step through them is physically longer — the *opacity* bar is
what's held constant across cascades, not the density bar. A single density threshold applied
to all cascades is therefore **wrong for every cascade but the base one** — it over-prunes the
outer (background) cascades that an unbounded 360° scene depends on.

### 2.2 The `min(0.01, mean)` bootstrap

At initialization the network outputs a near-constant small density (≈ $e^{-\text{bias}}$), so
*every* cell's optical thickness sits below 0.01. A hard `> 0.01` would mark the entire grid
empty → no samples survive → no gradients → it never starts training (we saw exactly this:
pure-white output, 421 steps/s, PSNR 6.6).

`min(0.01, mean)` fixes it: while the field is weak, `mean < 0.01`, so the threshold *is* the
mean and the grid keeps roughly its densest half — enough to bootstrap. As training sharpens
the field the mean climbs, the threshold saturates at 0.01, and the grid converges to "real
opacity ≥ 0.01." It is a self-annealing floor, not a magic number.

---

## 2.5 The core divergence: sampling is decoupled from the grid (why `1024`, not `128`)

This is the deepest structural difference between the two engines, and it is exactly the thing
the `√3/1024` constant exposes. **Instant-NGP's grid resolution and its sample spacing are two
different numbers on purpose.** Ours are the same number.

### 2.5.1 Where `1024` actually comes from

The intuition "unit cube at `128³` ⟹ step should be `1/128`" is correct **for our marcher** — we
walk the DDA one voxel at a time, so our step *is* the voxel size and we emit **one sample per
voxel**. Instant-NGP does not do this. Decompose their constant:

$$ \delta_{\min}=\frac{\sqrt3}{1024}=\frac{\sqrt3}{128\cdot 8}
   =\frac{1}{8}\cdot\underbrace{\frac{\sqrt3}{128}}_{\text{grid-cell diagonal}}
   =\frac{\text{cell diagonal}}{8}. $$

So **`1024 = 128 × 8`**: the grid resolution (`128`) times an **8× oversampling factor**, and the
`√3` is just the grid cell's *diagonal* (the worst-case path length through one cell). Read
plainly: Instant-NGP places **~8 samples across every occupied grid cell**, not one. Per axis the
ratio is `(1/128)/(√3/1024) = 8/√3 ≈ 4.6`; along the cell diagonal it is exactly `8`. The `1024`
is a **sampling** density deliberately chosen finer than the `128` **grid** density — they were
never meant to be equal.

### 2.5.2 The grid is a *skip structure*, not the sampling lattice

Their actual marching loop (`generate_training_samples_nerf`, paraphrased from `testbed_nerf.cu`
+ `nerf_device.cuh` — names simplified):

```cpp
while (aabb.contains(pos = o + t*d) && j < NERF_STEPS() /* 1024 */) {
    float    dt  = calc_dt(t, cone_angle);        // FINE step  (≈ cell_diag/8 near camera)
    uint32_t mip = mip_from_dt(dt, pos, max_mip); // coarsest cascade whose cell ≥ dt
    if (density_grid_occupied_at(pos, grid, mip)) {
        ++j; t += dt;                             // OCCUPIED: emit a sample, advance ONE fine dt
    } else {
        t = advance_to_next_voxel(t, cone_angle, pos, d, idir, mip); // EMPTY: DDA-skip to next cell
    }
}
```

Two regimes, and this split is the whole point:

- **Occupied space → fine stepping.** `t += dt` with `dt ≈ cell_diag/8`. Several samples
  (~5–8) land *inside* each occupied cell. The grid bit only said "something is here"; the
  integration is carried out at 8× finer resolution than the grid.
- **Empty space → DDA skip.** `advance_to_next_voxel` jumps straight to the next cell boundary
  (`distance_to_next_voxel` returns `min(tx,ty,tz)/res`), placing **no** samples.
  `if_unoccupied_advance_to_next_occupied_voxel` even climbs to coarser mips to skip the
  *largest* empty region in one jump.

So the `128³` grid is consulted **only to decide where to skip**; sample placement is governed by
the independent fine step. `mip_from_dt` keeps the two consistent: as `dt` grows with distance it
picks a coarser cascade whose cells are ≥ `dt`, so skip granularity always tracks step size.

### 2.5.3 Ours: the grid *is* the sampling lattice

Our `march_rays_dda_kernel` walks the DDA and emits **exactly one sample per occupied
finest-cascade voxel**, with `δ =` voxel size. The grid plays *both* roles simultaneously: it
skips empty voxels (good) **and** it fixes the sample spacing (limiting). There is no sub-cell
sampling — integration fidelity is welded to grid resolution.

| | Instant-NGP | ours |
|---|---|---|
| grid's role | empty-space **skip only** | skip **+** sampling lattice |
| samples / occupied cell | ~5–8 (fine `dt`) | **1** |
| step size | `dt = clamp(c·t, …)`, grid-independent | `δ =` voxel size |
| to sample 8× finer | lower one constant (`NERF_STEPS`) | need an 8× finer **grid** → 512× memory |

### 2.5.4 Why it matters (impact)

1. **Integration accuracy.** Opacity is `α = 1 − e^{−σδ}`. With our `δ =` a whole voxel, that's a
   *single rectangle* spanning the cell — we cannot resolve *where* in the voxel a surface lies,
   and any surface thinner than a voxel is smeared across the entire cell. Instant-NGP's small
   `δ` is a fine Riemann sum → accurate transmittance, crisp surfaces, less banding/aliasing.
   This is very likely part of why our real-scene PSNR plateaus where it does — coarse quadrature
   is a hard quality ceiling.
2. **Memory.** Decoupling is precisely what buys them *both* a cheap skip grid *and* fine
   sampling. Matching their fidelity under our one-sample-per-cell rule would require a ~`1024³`
   grid — `512×` the cells per cascade. Infeasible. They get the fidelity for free because
   samples don't consume grid memory.
3. **Gradient density.** One sample per voxel per ray = one density/color gradient per voxel;
   fine sampling gives denser, better-localized supervision → sharper, faster convergence.

### 2.5.5 How to close the gap (our single highest-impact change)

Make the grid **skip-only** and take **K sub-samples per occupied voxel**: in
`march_rays_dda_kernel`, when a voxel is occupied, emit `K` (≈4–8) samples at `δ = voxel/K`
instead of one, and keep skipping empty voxels exactly as now. The `128³` grid is untouched
(memory unchanged); only *occupied* space costs more MLP queries — and our fused MLP is fast
(<10 ms). This is a **larger** quality lever than Option A: Option A fixes cascade *correctness*,
sub-voxel sampling fixes integration *fidelity*. They are complementary, and together they are
the two things between us and Instant-NGP-class real-scene quality.

---

## 3. How this maps onto **our** engine

### 3.1 Stepping — we already do a quantized version

Our marcher (`processRays.cu:96 march_rays_dda_kernel`) is a **DDA, one sample per occupied
finest-cascade voxel**. Crucially, the step is the *local cascade's* voxel size, not a global
constant (`processRays.cu:138-153`):

```cpp
int cascade           = get_cascade(current_pos, aabb_min, aabb_max, num_cascades); // ≈ log2(dist)
float cascade_scale   = ...2^cascade...;
float3 c_aabb_extent  = aabb_extent * cascade_scale;
float3 voxel_size_l   = c_aabb_extent / res_l;     // step ≈ base_voxel · 2^cascade
```

Because `get_cascade` ≈ $\lceil\log_2(\text{distance})\rceil$ (`processRays.cu:11`), our step
size **doubles each cascade ≈ grows with distance** — i.e. we already get *geometric* step
growth, just **quantized to powers of two** at cascade boundaries instead of the smooth
$\delta=c\,t$. So exponential **step growth** is a *refinement* of what we have, not a missing
feature — but note the deeper gap is sample *density*, not step growth (we take **one** sample per
occupied voxel where Instant-NGP takes ~8; see §2.5, the bigger quality lever):

- continuous `δ=c·t` would decouple sample spacing from grid resolution and give a tunable
  `cone_angle` (smoother far-field, fewer samples), but
- the **bigger, cheaper win is making the occupancy threshold cascade-aware** so our existing
  per-cascade steps are matched by per-cascade density thresholds. That's §3.2.

(Note: our contraction `contract_pos` is applied to *every* MLP query (`processRays.cu:331`),
which already warps far content into the outer shells — another reason the per-cascade
occupancy threshold, not the step rule, is where the real-world fog/background tradeoff lives.)

### 3.2 Occupancy threshold — where we are vs. Instant-NGP

Today (`otherKernels.cu`): `updateTmpGrid` stores **raw σ** (`atomicMaxFloatFast(&tmpGrid[cell],
sigma)`), and `updateOccupancyGrid` thresholds with

```cpp
float threshold = fminf(0.43f, mean_density);   // raw-σ space
```

The `0.43` is the correct *base-cascade* equivalent of Instant-NGP's `0.01`, derived exactly
the same way as §2.1 but with **our** base step (the ±1.5 cube at $128^3$ → base voxel
$3/128 = 0.02344$):

$$ \sigma_{\text{thresh}}=\frac{0.01}{\Delta t_{\text{base}}}=\frac{0.01}{0.02344}\approx 0.427\approx 0.43 . $$

And we already use the **`min(0.43, mean)` bootstrap** (matching §2.2 — verified: fixes the
init cull, fog visibly reduced on house1). Good. **The one remaining gap is cascade scaling:**
a single `0.43` is correct only at cascade 0 and over-prunes outer cascades — exactly the
single-grid-vs-multi-cascade caveat that matters for bicycle ($K{=}4$).

**Option A (cascade-correct, ≈10 lines)** — adopt Instant-NGP's representation directly:

1. In `updateTmpGrid`, store **optical thickness** instead of σ:
   `cascade = cell_idx / G;  Δt = base_voxel · 2^cascade;  atomicMax(&tmpGrid[cell], σ·Δt);`
2. In `updateOccupancyGrid`, threshold `min(0.01, mean)` — now `0.01` is opacity directly, and
   because the stored value already carries `2^cascade`, **one threshold is correct for all
   cascades**. Outer-cascade background (big steps → opacity accrues at low σ) survives; inner
   fog (low σ, small steps) is pruned.

This is the cleanest form and it's what bicycle needs before the multi-cascade run.

### 3.3 Cross-reference table

| concept | Instant-NGP | ours |
|---|---|---|
| grid resolution | $128^3$ | `gridResolution` ($128^3$) |
| base step $\delta_{\min}$ | $\sqrt3/1024\approx1.69\text{e-}3$ (unit cube) | `base_voxel = aabb_extent/res` ($\approx0.0234$ for ±1.5) |
| per-cascade step | $\delta_{\min}\cdot 2^{\text{level}}$ | `voxel_size_l = base_voxel · 2^cascade` (DDA, implicit) |
| stepping | continuous `δ=clamp(c·t,δ_min,δ_max)` | DDA per voxel → **power-of-2-quantized** `δ` |
| stored grid value | optical thickness $\sigma\,\Delta t_{\text{level}}$ | **raw σ** today → **σ·Δt** under Option A |
| keep threshold | `min(0.01, mean)` opacity | `min(0.43, mean)` raw-σ today → `min(0.01, mean)` under Option A |
| base density thresh | $0.01\cdot1024/\sqrt3\approx5.92$ | $0.01/0.0234\approx0.43$ |
| decay | 0.95 / 16 iters | `decayValue` EMA in `updateMasterGrid` |
| update sampling | $M$ random cells, max-combine | per-sample `atomicMax` into `tmpGrid` |
| bit layout | Morton z-curve, DDA | Morton (`morton3d`, `processRays.cu:187`) |

---

## 4. Takeaways

1. **Exponential stepping = step proportionally to distance**, $\delta=\operatorname{clamp}(c\,t,
   \delta_{\min},\delta_{\max})$; the `to/from_stepping_space` warp is just the exact,
   invertible, integer-indexable form of that clamp. `cone_angle=0` collapses it to uniform
   stepping (synthetic scenes).
2. **We already get geometric step growth** through the cascaded DDA (step $\propto 2^{\text{cascade}}
   \propto$ distance), quantized to powers of two. Continuous exponential stepping is a polish
   item, not a blocker. But sample **density** is a different, bigger gap (§2.5): we place
   **one** sample per occupied voxel where Instant-NGP places ~8 — their grid is a skip
   structure (`1024 = 128×8` oversampling), ours is the sampling lattice. **Sub-voxel sampling
   (K samples per occupied voxel) is our largest untapped quality lever**, complementary to
   Option A.
3. **The `0.01` is an opacity threshold on $\sigma\,\Delta t$, not a density.** Storing
   optical thickness makes one threshold correct across all cascades; storing raw σ forces a
   per-cascade constant and a single value over-prunes outer cascades.
4. **`min(threshold, mean)` is a bootstrap**, not a hack: it prevents the init-time total cull
   and self-anneals to the true opacity floor.
5. **Next concrete step (Option A):** store $\sigma\,\Delta t_{\text{cascade}}$ in the grid and
   threshold `min(0.01, mean)` — makes our occupancy cascade-correct ahead of the bicycle
   ($K{=}4$) run. See also [`distortion_loss_gradient.md`](./distortion_loss_gradient.md) for
   the complementary lever (grid prunes low-density haze; distortion loss removes dense
   floaters).

---

## References

- Müller, Evans, Schied, Keller. *Instant Neural Graphics Primitives with a Multiresolution
  Hash Encoding.* SIGGRAPH 2022. Appendix E.1 (ray marching step size), E.2 (occupancy grids).
  <https://arxiv.org/pdf/2201.05989>
- NVlabs/instant-ngp source: `include/neural-graphics-primitives/nerf_device.cuh`
  (`calc_dt`, `to/from_stepping_space`, `MIN/MAX_CONE_STEPSIZE`, `NERF_MIN_OPTICAL_THICKNESS`,
  cascade/mip helpers) and `src/testbed_nerf.cu`
  (`splat_grid_samples_nerf_max_nearest_neighbor`, `grid_to_bitfield`,
  `update_density_grid_mean`). <https://github.com/NVlabs/instant-ngp>
- nerfstudio Instant-NGP notes: <https://docs.nerf.studio/nerfology/methods/instant_ngp.html>
