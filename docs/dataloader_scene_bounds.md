# DataLoader Scene-Bounds Derivation — Design & Mathematics

**Status:** design draft for review.
**Goal:** make the `DataLoader` ingest a `transforms.json` and derive, from the camera
poses alone, (1) a scene **center**, (2) an isotropic normalization **scale**, (3) a per-axis
**AABB**, and (4) a sensible **`num_cascades`** — falling back to the current `±1.5` cube when
the data doesn't justify anything tighter.

This must live in the **DataLoader**, not `colmap2nerf.py`, because **pre-built datasets
(DroneDeploy, instant-ngp, nerfstudio, Blender synthetic, …) never run our COLMAP script** —
they arrive already-posed, often in a different normalization convention, and frequently with
no point cloud. The loader is the one place every dataset passes through.

---

## 1. Motivation

### 1.1 The fog problem
NeRF only learns geometry where rays are *observed*. Any trainable volume inside the AABB that
no camera constrains will fill with semi-transparent **fog / floaters**, because the network can
lower training loss by parking density there (especially using view-dependent color to satisfy
each training view). The primary fix is structural: **shrink the modeled volume to the observed
volume** so there is no unconstrained space to hallucinate into (§4–6). A complementary
training-time regularizer — the mip-NeRF 360 **distortion loss** — handles the haze that survives
*inside* the observed volume (§12). Together with the existing random-background augmentation they
form a three-legged defense (§12.6).

### 1.2 The `house1` failure (worked motivation)
The DroneDeploy "big house" dataset (`data/house1`) is a **nadir drone capture**:

- 50 cameras, all pointing nearly straight **down** (`forward.z ≈ −0.91`).
- All at essentially the **same altitude** (camera `Z` spread `≈ 0.06` vs `XY` spread `≈ 1.2`).
- Tagged `aabb_scale: 1` → the author declares it a **bounded** scene.
- Format is **instant-ngp** (`fl_x`, `cx`, `k1..k6`, `aabb_scale`), *not* our `colmap2nerf`
  output, and it carries **no `num_cascades`**.

Setting `num_cascades = 8` on this scene produced "mostly fog with a tiny blurry house."
Why:

- Cascade `c` covers extent `~1.5·2^c`, so 8 cascades model a half-extent of
  `1.5·2^7 ≈ 192` units, while the real content sits within `~0.6` units.
- ~99.99% of the modeled volume is empty and unobserved → fog.
- The house gets a microscopic share of a fixed grid budget → blur.

Plus two structural issues this design addresses:

- **Off-center & small.** The loader doesn't normalize; instant-ngp's normalization placed the
  *cameras* near the origin at `z ≈ 0.24` with the ground *below* them. The content is offset
  and tiny inside the `±1.5` cube, and the empty volume **above** the cameras becomes a fog slab.
- **Low vertical parallax.** All cameras at one altitude looking down ⇒ depth along the view
  direction is barely triangulated ⇒ the network smears density vertically (fog column). This is
  intrinsic to the data; bounds can't fully fix it, but a thin-Z box stops it from filling the
  sky.

**Lesson:** physically large ≠ unbounded. Cascades are for background receding toward infinity;
this scene needed *better placement and tighter bounds*, not more cascades.

---

## 2. What `transforms.json` actually gives us

From each frame's c2w (4×4), in the engine's OpenGL convention (camera looks down −Z):

| Symbol | Meaning | Extraction |
|---|---|---|
| `Cᵢ` | camera center (world) | translation column `col3` |
| `fᵢ` | forward / view dir | `−col2` |
| `rᵢ` | right axis | `col0` |
| `uᵢ` | up axis | `col1` |
| `hx, hy` | half-FOV (horiz/vert) | `camera_angle_x/2`, `camera_angle_y/2` |

What we **do not** have: any 3D geometry (no point cloud, no depth). This drives a clean split
of what's recoverable:

| Quantity | From poses? | Why |
|---|---|---|
| Scene **center** | ✅ robust | rays converge on what was photographed |
| **Scale / AABB** | ⚠️ up to a framing/depth assumption | poses have a global scale ambiguity |
| **`num_cascades`** | ❌ not reliably | cascades measure how far *content* extends; far cameras ≠ big scene |

---

## 3. Hard constraints imposed by the engine

These were read directly from the kernels and **dictate** parts of the algorithm.

### 3.1 Per-axis (rectangular) AABB is natively supported
- `contract_pos` (`processRays.cu:21`): normalizes **per axis**,
  `(pos.a − aabb_min.a) / aabb_extent.a`.
- `cellSize` (`otherKernels.cu:629`): `(c_aabb_max.a − c_aabb_min.a) / gridResolution.a`,
  per axis.
- `get_cascade` (`processRays.cu:11`): per-axis, per-side ratios.

⇒ We can pass independent per-axis (even asymmetric) bounds **without touching the kernels**.
Only `Pipeline.cu:111-112` (the hardcoded `±1.5`) needs to read from the loader instead.

### 3.2 Cascades are anchored at the **origin** ⇒ content must be recentered
```
c_aabb_min = aabb_min * exp2(cascade)      // otherKernels.cu:626
c_aabb_max = aabb_max * exp2(cascade)
get_cascade: max over axes of  pos.a / aabb_max.a  (or / aabb_min.a)   // processRays.cu:12-15
```
Cascades scale the box about the **origin**, and `get_cascade` measures distance from the origin
as a fraction of the boundary. So nested cascades are concentric about the origin, **not** about
the box center. **Recentering the content to the origin is mandatory**, or cascade 0 won't
contain it and points spill into higher cascades for no reason.

### 3.3 Scale must be isotropic; only the box may be anisotropic
A single scalar `s` multiplies all translations. We may **clip** the box per axis, but we must
**not** scale per axis — anisotropic scaling shears ray directions and distorts the density /
hash-grid isometry. **One scale, three box half-extents.**

---

## 4. Centering — least-squares ray convergence

The object is wherever the cameras are *looking*, not where they *are*. (Camera **centroid** is
the naive choice and only coincides with the look-point for a symmetric full orbit; for a partial
arc or uneven distances it's biased — see §8.) Each camera is a ray `(Cᵢ, fᵢ)`. Find `p`
minimizing the sum of squared perpendicular distances to all rays.

Perpendicular component of `(p − Cᵢ)` w.r.t. unit `fᵢ`:

$$(p - C_i) - \big((p - C_i)\cdot f_i\big) f_i = A_i (p - C_i), \qquad A_i = I - f_i f_i^{\top}$$

(`Aᵢ` is the projector onto the plane ⟂ `fᵢ`.) Minimize
$\sum_i (p - C_i)^{\top} A_i (p - C_i)$; setting the gradient to zero gives a single 3×3 system:

$$\Big(\underbrace{\textstyle\sum_i A_i}_{M}\Big)\, p = \underbrace{\textstyle\sum_i A_i C_i}_{b}, \qquad p^\* = M^{-1} b$$

```
M = 0 (3x3);  b = 0 (3)
for each frame:
    f = normalize(-col2);  C = col3
    A = I - f·fᵀ
    M += A;  b += A·C
center = solve(M, b)        # 3x3 (add λI for safety)
```

**Degeneracy guard.** If all forwards are parallel (forward-facing / nadir), `M` is
rank-deficient, `det(M) → 0`, and the depth coordinate of `p*` is meaningless. Detect via
`det(M)` (or condition number); on failure fall back to the camera centroid for the in-plane
coordinates and rely on the frustum-march box (§6) for the depth axis. Add `λI` (small Tikhonov
term) for numerical safety regardless.

This is the `central_point` trick from instant-ngp; robust for any orbit / inward-looking capture.

---

## 5. Scale — breaking the pose scale ambiguity

Poses can't distinguish a small object shot up close from a big one shot far away (global scale
is unobservable). One extra assumption is required. Two independent estimators, cross-checked:

**(a) Framing estimator (uses FOV).** Photographers frame the subject to fill the frame, so its
angular radius ≈ a fixed fraction of the FOV. With `dᵢ = ‖Cᵢ − p*‖`:

$$r_{\text{obj}} \approx \operatorname{median}_i(d_i)\cdot \tan(h_x)\cdot f_{\text{fill}}, \qquad f_{\text{fill}} \approx 0.5$$

**(b) Distance-shell bound.** The subject sits in front of the nearest camera, so
`r_obj ≤ Q₁₀(dᵢ)` (low percentile, robust to one stray close pose).

Take `r_obj = clamp(framing_estimate, small, Q₁₀(dᵢ))`.

> For object-centric scenes this alone suffices: `s = 1/r_obj`, recenter by `−p*`, and the
> object lands in the unit sphere inside the `±1.5` box. For anisotropic captures (drone slab,
> forward-facing) the **frustum-march box of §6 supersedes this** and produces the scale via its
> largest axis. The framing estimate is then a sanity cross-check.

---

## 6. Per-axis AABB — sample the observed volume

The box should enclose the volume the frustums actually sweep, and nothing more.

### 6.1 Frustum-corner rays
Per camera, the 4 corner directions plus the center:

$$d_{s_x,s_y} = \operatorname{normalize}\big(f_i + \tan(h_x)\,s_x\, r_i + \tan(h_y)\,s_y\, u_i\big), \quad s_x,s_y\in\{-1,+1\}$$

### 6.2 March to a bounded depth
Sample along each ray at depths `t ∈ (t_near, D]`, plus include the camera centers themselves:

$$P = C_i + t\, d_{s_x,s_y}$$

The march depth `D` is the one ambiguous knob (the scale ambiguity in disguise), set by regime
(§7):

- **Orbit** (`inwardness ≳ 0.9`, rays converge): `D ≈ 2·median‖Cᵢ − p*‖` — march to/past center.
- **Nadir / parallel** (`inwardness ≈ 0`, `det(M)` small): rays don't triangulate depth, so
  anchor on the rig footprint — `D ≈ k · (lateral camera spread)`, `k ≈ 1.5–2`. For aerial
  mapping the captured ground depth scales with the footprint, which *is* observable.

### 6.3 Robust per-axis extents
For each axis `a`, reject the far-march tail and stray poses with percentiles:

$$\text{lo}[a] = Q_{2\%}(\{P_a\}), \qquad \text{hi}[a] = Q_{98\%}(\{P_a\})$$

yielding an (asymmetric) box `[lo, hi]`.

### 6.4 Recenter, isotropic normalize, clamp to the 1.5 default
```
c[a]    = (lo[a] + hi[a]) / 2
half[a] = (hi[a] - lo[a]) / 2

translate all camera centers by  -c        # content → origin   (REQUIRED, §3.2)
s       = 1.5 / max_a(half[a])             # ONE isotropic scalar (§3.3)
scale all camera centers by  s

H[a]    = half[a] * s                       # per-axis half-extent; largest axis → 1.5
H[a]   *= 1.10                               # safety margin so grazing content isn't clipped
H[a]    = clamp(H[a], H_min, 1.5)            # never exceed base box; cascades cover beyond

aabbMin = (-H.x, -H.y, -H.z)
aabbMax = (+H.x, +H.y, +H.z)
```

`1.5` stays the canonical scale (so `BASE_HALF_EXTENT` and all cascade constants remain valid):
the **largest** observed axis normalizes to `1.5`; thinner axes deviate **below** it. Recentering
makes the box symmetric per axis, which keeps cascade nesting trivial (the math also supports
asymmetric if ever needed).

---

## 7. `num_cascades` — the convergence-based classifier

Cascades cover **content extending far beyond the subject** (background → infinity). That extent
is **not** measurable from poses (a figurine on a turntable inside a cathedral looks identical,
from poses, to one in a void). So we do **not** try to measure it; we classify the **capture
topology**, which *is* pose-observable, and attach a sensible default per regime.

Three signals (all from §4 quantities):

**(1) Convergence residual** — do the rays actually meet?

$$\bar\rho = \frac{\operatorname{median}_i \lVert A_i (p^\* - C_i)\rVert}{r_{\text{obj}}}$$

Small ⇒ rays stab a compact point ⇒ bounded object. Large ⇒ no shared focus.

**(2) Inwardness** — do cameras face the center or away?

$$\text{inward} = \frac{1}{N}\sum_i f_i \cdot \operatorname{normalize}(p^\* - C_i)$$

`+1` orbit (bounded) · `−1` panorama-from-inside (unbounded) · `0` forward/nadir parallel.

**(3) Forward spread** — the degeneracy detector.

$$\text{spread} = \operatorname{Var}_i(f_i)$$

`≈ 0` ⇒ parallel forwards ⇒ `M` near-singular ⇒ forward-facing/nadir regime. (Same test as the
§4 degeneracy guard — one geometric fact drives both the centering fallback and this branch.)

Note spread is **ambiguous alone**: a 360 orbit and an inside-out panorama both have high spread.
Combine with inwardness (high spread **+** inward = orbit; high spread **+** outward = panorama).

```
if spread < ε_low:                       regime = FORWARD/NADIR  → cascades = 5..6   # unbounded far field
elif inward > 0.9 and rho_bar < 0.15:    regime = OBJECT_CENTRIC → cascades = 1      # bounded
elif inward < -0.5:                      regime = PANORAMA       → cascades = 5..6   # unbounded
else:                                    regime = INTERMEDIATE   → cascades = 2..3
```

**This is a topology prior, not a measurement — log it as such.** A bounded object inside a huge
unobserved background still classifies as `OBJECT_CENTRIC → 1`, because no pose-only method can
see that background. For an accurate count, consume geometry when available (next section).

> **`house1` note:** `inwardness ≈ 0`, `aabb_scale = 1` ⇒ NADIR-but-bounded. The classifier's raw
> NADIR branch would say "unbounded → many cascades," which is wrong here — the scene is a bounded
> slab. The `aabb_scale` hint (when present) and the *finite* frustum-march box both indicate
> bounded, so **respect an explicit `aabb_scale`/`num_cascades` in the file over the classifier**,
> and treat the classifier as the last resort. Correct answer for `house1`: **1**.

---

## 8. Centroid vs. ray-convergence vs. point-cloud median (centering comparison)

| Method | Needs | Robust to | Fails when |
|---|---|---|---|
| Camera **centroid** | poses | nothing extra | partial arcs, uneven camera distance (biased toward the rig, not the look-point) |
| **Ray convergence** (§4) | poses | camera distribution / partial coverage | parallel forwards (depth coord meaningless) → guard + fallback |
| Point-cloud **median** | a point cloud | outliers (median) | sparse/failed reconstruction |

```
        cameras (90° arc)
        ●  ●  ●
       ●        ●
      ●          ← centroid lands HERE (on the arc)
                         ✕  ← object is HERE (where they all look) — convergence finds this
```

Order of preference: **point-cloud median (if available) > ray convergence > centroid (fallback)**.

---

## 9. `num_cascades` with geometry (preferred when points exist)

If the dataset carries a point cloud (`points3D.txt` or a json sidecar), skip the classifier and
use the real extent — this is exactly what `colmap2nerf.py` does:

$$\text{ratio} = \frac{\text{scene\_radius}}{\text{obj\_radius}}, \qquad
\texttt{num\_cascades} = \operatorname{clamp}\!\Big(\big\lceil \log_2(\text{ratio}/1.5)\big\rceil + 1,\ 1,\ 6\Big)$$

where `obj_radius` = 90th-percentile point distance from center (subject), `scene_radius` =
95th-percentile distance over **all** points. `BASE_HALF_EXTENT = 1.5` must match the box.

---

## 10. Full pipeline (summary)

```
INPUT: transforms.json  (poses; optional camera_angle_y, aabb_scale, num_cascades, points)

1.  Parse Cᵢ, fᵢ, rᵢ, uᵢ, hx, hy.
2.  center p* ← solve (Σ Aᵢ) p = Σ Aᵢ Cᵢ,  Aᵢ = I − fᵢfᵢᵀ      (centroid fallback if det small)
3.  regime signals: inwardness, residual ρ̄, spread, det(M).
4.  observed-volume box:
        frustum-corner rays → march to depth D (regime-dependent) → samples P
        lo[a],hi[a] ← Q2%,Q98% of {P_a}
5.  c=(lo+hi)/2; half=(hi-lo)/2
        recenter poses by −c                      # REQUIRED (cascades anchor at origin)
        s = 1.5 / max_a half[a]; scale poses by s  # ISOTROPIC
        H[a] = clamp(1.10 · half[a]·s, H_min, 1.5)
        aabbMin=−H, aabbMax=+H
6.  num_cascades:
        if file has explicit aabb_scale/num_cascades → respect it
        elif points available → log2-ratio formula (§9)
        else → regime classifier (§7), logged as an estimate
7.  FALLBACK: too few frames / degenerate / ill-conditioned → ±1.5 cube, num_cascades=1

OUTPUT: recentered+scaled poses, aabbMin/Max (per axis), num_cascades
```

---

## 11. Worked prediction — `house1`

- Cameras: XY `±0.6`, Z `0.27`; all looking down ⇒ NADIR; `inwardness ≈ 0`, `det(M)` ok but `p*`
  lands at the camera plane (`z ≈ 0.25`) → depth from frustum march, not from `p*`.
- `D ≈ 1.5 × 0.6 ≈ 0.9`. Frustum samples sweep a **wide, thin slab**: XY to `~±1.1`, Z from the
  camera altitude down to the marched ground.
- Step 5: `s ≈ 1.5 / 1.1 ≈ 1.35`; **`H ≈ (1.5, 1.5, 0.6)`**, content recentered to origin.
- `num_cascades`: respect `aabb_scale = 1` ⇒ **1**.

Effect: the empty sky volume above the house (`z` from `~0.25` to `1.5`) is **outside the box** —
no samples are drawn there, so it cannot hallucinate fog — and the `1.35×` scale-up sharpens the
house. The vertical-parallax smearing (§1.2) still wants regularization, but it can no longer fill
the sky.

---

## 12. Distortion loss (mip-NeRF 360) — the third leg against fog

Centering + scaling + tight bounds (§4–6) remove *unobserved* volume, so fog has nowhere to
form. Random background augmentation (already in `Pipeline.cu:195`) makes semi-transparent
density expensive by forcing it toward binary. The **distortion loss** from mip-NeRF 360
(Barron et al., 2022) attacks what those two miss: *observed-but-ambiguous* volume — translucent
haze and disconnected floaters spread **along a ray** that the photometric loss alone tolerates.
This is exactly `house1`'s low-vertical-parallax smearing (§1.2), which bounds cannot fix because
the smear lives inside the observed slab.

### 12.1 Idea
A ray's render is a set of weights `wᵢ = Tᵢ αᵢ` (the same weights computed in
`compositing.cu:46-55`). A clean surface puts **all** weight in one compact cluster; fog/floaters
**spread or split** the weight across distant depths. The distortion loss penalizes the *spread*
of the per-ray weight distribution, driving it to consolidate into a single compact mode and zero
elsewhere.

### 12.2 Definition
Let `sᵢ` be the **normalized** ray distance of sample `i` (map metric `t ∈ [near, far] → s ∈
[0,1]`; normalizing makes the loss scale-invariant — raw `t` would bias it by scene scale). With
piecewise-constant weights over intervals `[sᵢ, sᵢ₊₁]`, midpoints `mᵢ = (sᵢ + sᵢ₊₁)/2`, and
lengths `δᵢ = sᵢ₊₁ − sᵢ`:

$$\mathcal{L}_{\text{dist}} = \underbrace{\sum_{i,j} w_i w_j \,\lvert m_i - m_j\rvert}_{\text{bilinear}} \;+\; \underbrace{\tfrac{1}{3}\sum_i w_i^2\, \delta_i}_{\text{self}}$$

- **Bilinear term:** penalizes weight at *separated* depths → pulls all mass together; kills
  floaters split along the ray.
- **Self term:** penalizes weight spread *within* a wide interval → sharpens each contribution.

### 12.3 O(N) evaluation (avoid the N² double sum)
Naively the bilinear term is `O(N²)` per ray. But the `sᵢ` are already sorted along the ray, so
use inclusive prefix sums `W_{≤k} = Σ_{j≤k} wⱼ` and `(WS)_{≤k} = Σ_{j≤k} wⱼ mⱼ`, with totals
`W`, `WS`:

$$\mathcal{L}_{\text{bi}} = \sum_k w_k\Big[\, m_k\,(2 W_{\le k} - W) - \big(2 (WS)_{\le k} - WS\big)\Big]$$

(the `i = j` diagonal contributes `0`, so including it is harmless). One scan → `O(N)`.

### 12.4 Gradient (reuses the existing weight→σ backward)
`L_dist` is an explicit function of `(s, w)`; its gradient is closed-form and also `O(N)` from
the same prefix sums:

$$\frac{\partial \mathcal{L}_{\text{dist}}}{\partial w_k} = 2\Big[\, m_k\,(2 W_{\le k} - W) - \big(2 (WS)_{\le k} - WS\big)\Big] + \tfrac{2}{3}\, w_k\, \delta_k$$

This is just an **extra upstream gradient on the per-sample weights** `w_k`. The engine already
backprops `∂L_rgb/∂w_k → ∂L/∂σ_k` through the volume-rendering recurrence (`w_k = T_k α_k`, with
`T` coupled across samples) in `compositing.cu`. So integration = **add `∂L_dist/∂w_k` to that
same upstream and reuse the existing backward path** — no new σ-gradient derivation needed.

### 12.5 Integration into the engine
- We already have per-sample `wᵢ` (`compositing.cu:46-55`) and `tᵢ` (`d_sparse_ts`). Compute
  `sᵢ` by normalizing `tᵢ` over the ray's `[near, far]` (or the cascade-contracted distance, to
  stay consistent with `contract_pos`).
- New kernel: per ray, prefix-scan over its samples → `L_dist` and `∂L_dist/∂wᵢ`.
- Add `λ_dist · L_dist` to the loss; feed `∂L_dist/∂w` into the existing weight gradient.
- **`λ_dist`:** mip-NeRF 360 used `~0.01` relative to the data loss; start small (`1e-3 … 1e-2`)
  and tune.
- **Warmup:** the loss can collapse geometry if applied before a coarse surface forms. Enable it
  after a warmup (first `~10–20%` of steps) or ramp `λ_dist` up — ties into the cosine-LR
  schedule already in `Pipeline.cu:184-187`.

### 12.6 Why all three together
| Mechanism | Targets | Where |
|---|---|---|
| Centering + tight per-axis AABB (§4–6) | *unobserved* empty volume | DataLoader / bounds |
| Random background augmentation (existing) | semi-transparent vs. solid ambiguity | training, per-step bg |
| **Distortion loss** (§12) | *observed-but-ambiguous* haze & split floaters | training loss + backward |

Bounds give fog nowhere to live; random-bg makes it expensive; the distortion loss makes it
actively penalized. `house1`'s vertical smear needs the third leg.

---

## 13. Correctness notes & gotchas

1. **Isotropic scale, anisotropic box** (§3.3). One scalar `s`; three half-extents. Never scale
   per axis.
2. **Recenter to origin is mandatory** (§3.2), not cosmetic — cascade indexing depends on it.
3. **Rectangular cells.** Uniform `gridResolution` + thin-Z box ⇒ finer Z cells (good for thin
   content) but a different cell count per volume. Optionally make `gridResolution.a ∝ H[a]` to
   keep cells ~cubic; `cellSize` already supports it.
4. **Don't over-tighten.** Content poking past a tight box at `num_cascades=1` gets clamped by
   `contract_pos`. The `Q98%` percentile + `1.10` margin guard this — keep them.
5. **Respect explicit file hints.** `aabb_scale` / `num_cascades` in the json beat the pose-only
   classifier (the `house1` lesson). The classifier is a last resort.
6. **Points beat poses.** When a point cloud is present, use it for §6 and §9 — it removes the
   march-depth `D` ambiguity entirely. The frustum march is the pose-only fallback.
7. **Ownership.** Since pre-built datasets can't run `colmap2nerf.py`, the DataLoader owns
   centering+scaling. If a dataset *was* produced by our script (already normalized), the loader's
   derivation should be ~idempotent (center ≈ 0, max half ≈ 1) — but guard against
   double-normalization (e.g. skip if a `normalized: true` flag is present).

---

## 14. Open questions for review

- `f_fill`, `k` (march-depth multiple), percentile cutoffs, `H_min`, `ε_low`, `λ` — default
  values to pin down empirically.
- **Distortion loss** (§12): `λ_dist` magnitude and warmup schedule; and whether `sᵢ` should be
  normalized over per-ray `[near, far]` or the cascade-contracted distance (the latter keeps it
  consistent with `contract_pos` but spends resolution differently across cascades).
- Should the box stay **symmetric** (recenter to box center) or allow **asymmetric** (tight top
  for nadir, generous bottom)? Asymmetric is supported by the math and is *strictly* better for
  occluded-below cases, at the cost of slightly trickier cascade reasoning.
- Per-axis `gridResolution` vs. uniform (gotcha 3).
- How to carry the point cloud to the loader for pre-built datasets that have one (sidecar file?).
- Double-normalization guard for our own `colmap2nerf` outputs (gotcha 7).

---

## 15. Review addendum — §6 must be regime-gated (orbit "camera-shell" inflation)

**Finding (review of §6.2).** The march depth `D ≈ 2·median‖Cᵢ − p*‖` is wrong for
inward-converging (orbit) captures — and the bug is the `2·d` factor and the union-of-frustums
geometry, *not* the choice of statistic.

### 15.1 Why the orbit march inflates to the camera shell
Marching from each camera produces samples in the **empty space between the camera and the
object**. For an inward orbit, rays arrive from all directions and crisscross, so the union of
samples fills the entire interior **ball of radius ≈ camera distance**, not the object.

> **Worked example.** 100 cameras on a ring of radius `d = 3` looking at an object of radius
> `r_obj = 1` at the origin, `D = 2d = 6`. Camera `(3,0,0)` marches to `(−3,0,0)`; `(0,3,0)`
> marches to `(0,−3,0)`; etc. The samples fill the **disk of radius 3**, so `Q98%` per axis ≈ `3`
> → the box half-extent is the *camera radius*, not the object.

After normalizing the largest axis to `1.5`, the object (radius 1) lands at radius
`1.5·(1/3) = 0.5` — it fills only a third of the box linearly, **~1/27 by volume**. That is the
"object underfills the grid → blur" failure from the scale discussion. (It does *not* cause fog
here — rays pass through the gap and constrain it empty — it wastes resolution.)

**Consequence:** switching `median → Q95` makes this *slightly worse*, not better, because it
nudges `D`/`d` upward. The statistic was never the problem.

### 15.2 The frustum march is only valid for diverging / parallel captures
For nadir / forward-facing / panorama captures, rays **don't** crisscross, so the union of
frustums genuinely is the content slab — which is why the march works for `house1`. The
distinction is exactly:

| Capture | Rays | Union of frustums | Frustum march? |
|---|---|---|---|
| Orbit / inward (`inwardness ≳ 0.9`) | converge | whole interior ball (≫ object) | ❌ inflates → use `r_obj` |
| Nadir / forward / panorama | diverge or parallel | the content slab | ✅ correct |

### 15.3 Fix — regime-gate the box construction (amends §6.2 / §6.4)
- **Orbit / inward-converging:** do **not** march. Use §5's framing-estimator `r_obj` directly
  for a symmetric box (`H[a] = r_obj`, normalized so the max axis → `1.5`). This restores the
  object to unit radius and maximizes resolution.
- **Nadir / forward / panorama (diverging or parallel):** use the frustum march, with
  **`D = k · Q95(footprint)`** — i.e. the `median → Q95` change from review, scoped to the only
  branch where the march applies. `Q95` (not `max`) rejects outlier poses; `Q98%` in §6.3 still
  trims the sample tail.

### 15.4 Net change to §6
```
if regime == ORBIT (inwardness ≳ 0.9):
    H[a] = r_obj              # from §5 framing estimator; symmetric box, no march
else:  # NADIR / FORWARD / PANORAMA
    D    = k · Q95(footprint) # was 2·median camera distance — fixed
    march frustum-corner rays to D  →  samples P
    lo[a],hi[a] = Q2%, Q98% of {P_a}
    H[a] = (hi[a]-lo[a]) / 2
# then §6.4 recenter + isotropic scale + clamp to 1.5 as before
```

**Takeaway:** the frustum march measures *where cameras can see*, which equals *where content is*
only when rays don't converge. For orbits that's false (cameras see the whole interior), so the
object's scale must come from `r_obj`, not from marching to the opposite camera.

---

## 16. Refinement — DataLoader-owned COLMAP point cloud (preferred path)

This supersedes the pose-only derivation (§4–7, §15) **whenever a point cloud is available**.
The pose-only path is retained only as the fallback when COLMAP fails or returns too few points.

### 16.0 Resolved open questions (from §14)
- **Asymmetric box: ADOPTED.** The AABB may have `aabb_min.a ≠ −aabb_max.a` when the content
  wants it (e.g. a house pokes *up* from a flat ground plane). The kernels already support this
  (per-axis, per-side `get_cascade`; per-axis `contract_pos`/`cellSize`). The only requirement is
  that the **origin stays inside the content** (recenter to the subject center) so cascade 0
  contains the subject.
- **Grid resolution: FIXED.** `gridResolution` stays uniform; **cascades scale the box, the
  resolution fits it** (constant cells per cascade level). Consequence: with an anisotropic box
  the cells are rectangular and the thin axis gets *finer* sampling per unit length — accepted
  (it's free resolution where content is thin). We do **not** make `gridResolution.a ∝ H[a]`.

### 16.1 Architecture / data flow
The main script only **extracts images**. The DataLoader owns geometry:

```
DataLoader(images_dir):
    if transforms_train.json exists:
        poses ← json (TRUSTED — may be synthetic ground truth / higher quality than COLMAP)
        if points cache exists: points ← cache
        else:
            run COLMAP (feature → match → map)          # for the POINT CLOUD only
            register COLMAP frame → json frame (§16.2)   # poses differ; must align
            points ← similarity-transformed COLMAP points
            save points cache (.ply / bin sidecar)       # COLMAP runs ONCE
    else:                                                # pure images (no json)
        run COLMAP (or reuse cache) → poses AND points   # same frame, no registration
    if points are sufficient (≥ N_min): geometric path (§16.3–16.6)
    else:                                                 # COLMAP sparse/failed
        pose-only fallback (§4–7, §15)
```

### 16.2 Registration — align COLMAP points to the trusted poses (Umeyama similarity)
COLMAP reconstructs its **own** camera poses `{Aᵢ}` (centers) *and* points `{Xⱼ}` in an arbitrary
frame. If we keep the json poses `{Bᵢ}`, the points are in the wrong frame. Because COLMAP also
posed the **same images**, we have center correspondences `Aᵢ ↔ Bᵢ` (matched by filename) and can
solve the 7-DoF similarity `(s_reg, R, t)` that maps COLMAP → json:

$$\min_{s_{\text{reg}},R,t}\ \sum_i \big\lVert B_i - (s_{\text{reg}} R A_i + t)\big\rVert^2$$

**Umeyama (1991) closed form.** With `μ_A, μ_B` the centroids, `σ_A² = (1/n)Σ‖Aᵢ − μ_A‖²`, and
cross-covariance `Σ = (1/n) Σ (Bᵢ − μ_B)(Aᵢ − μ_A)^⊤`, take the SVD `Σ = U D V^⊤`:

$$S = \begin{cases} I & \det(U)\det(V) \ge 0\\ \operatorname{diag}(1,1,-1) & \text{otherwise}\end{cases}, \quad
R = U S V^\top, \quad s_{\text{reg}} = \frac{\operatorname{tr}(D S)}{\sigma_A^2}, \quad t = \mu_B - s_{\text{reg}} R\,\mu_A$$

Then transform every point: `Xⱼ^json = s_reg R Xⱼ^colmap + t`.

> **`s_reg` is the *registration* scale (COLMAP→json units) — a different quantity from the
> *normalization* scale `s_norm` of §16.4.** Keep them distinct in code.

**Caveats:**
- Needs `≥ 3` non-collinear correspondences. The `S` term fixes the reflection/handedness
  ambiguity (e.g. an OpenCV↔OpenGL world flip).
- **Coplanar cameras** (nadir `house1`: all at one altitude) → the in-plane fit is well-posed but
  the out-of-plane direction is weakly constrained; check the **residual** `Σ‖Bᵢ − (s_regRAᵢ+t)‖²`.
- If the residual is large, the json convention differs by something the proper-rotation fit
  can't absorb → log and fall back to pose-only (§16.1).

### 16.3 Subject vs. background, center, per-axis extents
Each `points3D` entry carries a **track** = list of `(image_id, point2d_idx)`; its length is the
**view count** `vⱼ` (how many cameras saw it). Well-observed points lie on the main subject;
stray background/floaters are seen rarely.

```
subject  = points with vⱼ in the top 20% by view count    # isolate the object
center   = component-wise MEDIAN of subject               # robust to floaters; origin goes here
```

Recenter all poses **and** points by `−center` (origin → subject center; §16.0 requirement).
Then per axis `a`, per side, take **robust percentile** extents of the subject (reject the
outer-tail floaters that survived the view-count filter):

$$e_a^{+} = Q_{90\%}\big(\{X_{j,a} : X_{j,a} > 0\}\big), \qquad
  e_a^{-} = \big\lvert Q_{10\%}\big(\{X_{j,a} : X_{j,a} < 0\}\big)\big\rvert$$

(The two sides differ ⇒ the **asymmetry the content wants**, e.g. taller `+z` than `−z`.)

### 16.4 Isotropic normalization scale
Map the **widest subject extent** to radius `1.0`, deliberately leaving the `1.0 → 1.5` band of
the base box as headroom for the outer-10% subject points and immediate background (so they stay
in cascade 0 instead of spilling to cascade 1):

$$s_{\text{norm}} = \frac{1.0}{\max_a\big(\max(e_a^{+}, e_a^{-})\big)}$$

Apply `s_norm` (one **isotropic** scalar — §3.3) to all pose translations and points.

> *Difference from the pose-only §6.4 (which maps to `1.5`):* there we had no subject/background
> separation, so we filled the box. Here the point cloud *separates* them, so we reserve the
> headroom on purpose and let the background drive cascades (§16.5). Same philosophy, better data.

### 16.5 `num_cascades` from real geometry (replaces the §7 classifier)
With the subject at radius `~1.0`, measure how far the **whole** cloud reaches:

$$\text{scene\_radius} = Q_{95\%}\big(\{\lVert X_j\rVert_{\text{norm}}\}\big), \qquad
\texttt{num\_cascades} = \operatorname{clamp}\!\Big(\big\lceil \log_2(\text{scene\_radius}/1.5)\big\rceil + 1,\ 1,\ 6\Big)$$

This is the §9 formula, now always available — it *measures* background extent instead of
*guessing the regime*. A file-supplied `aabb_scale`/`num_cascades` still overrides (the `house1`
lesson, §7).

### 16.6 AABB assembly (asymmetric, per-axis, fixed grid)
```
H_a^+ = clamp( margin · e_a^+ · s_norm , H_min , 1.5 )      # margin ≈ 1.1–1.5
H_a^- = clamp( margin · e_a^- · s_norm , H_min , 1.5 )
aabb_min = ( -H_x^- , -H_y^- , -H_z^- )                     # asymmetric allowed
aabb_max = ( +H_x^+ , +H_y^+ , +H_z^+ )
gridResolution = fixed uniform uint3                        # §16.0: grid scales, res fits
```

### 16.7 Worked prediction — `house1` with points
- COLMAP runs on the 50 images → dense-ish ground + house points; registered to the json poses
  via §16.2 (`s_reg` ~ matches the instant-ngp normalization scale; coplanar-camera residual
  checked).
- Subject (top-20% viewed) = the house + nearby ground. Median center sits **on the ground**, not
  at the camera plane — fixing the `p* ≈ camera altitude` problem (§11) *with geometry instead of
  the march heuristic*.
- Per-axis extents: wide XY, thin Z, and **`+z` (house height) > `−z` (flat ground)** → genuinely
  asymmetric box. After `s_norm`: roughly `aabb_max ≈ (1.5, 1.5, +0.5)`,
  `aabb_min ≈ (−1.5, −1.5, −0.15)`.
- `num_cascades`: measured from the actual ground extent (likely `1`, consistent with
  `aabb_scale=1`) rather than the nadir classifier's misleading "unbounded → many."

### 16.8 What stays vs. what becomes fallback
| Concern | With point cloud (preferred) | Pose-only fallback |
|---|---|---|
| Center | subject **median** (§16.3) | ray convergence `p*` / box midpoint (§4, §6.4) |
| Scale | widest subject extent → `1.0` (§16.4) | framing estimator / frustum march (§5, §6) |
| Per-axis AABB | subject percentiles, asymmetric (§16.6) | frustum march, regime-gated (§15) |
| `num_cascades` | measured `log2`-ratio (§16.5) | regime classifier, logged estimate (§7) |
| Registration | Umeyama if json poses kept (§16.2) | n/a (COLMAP poses used directly) |

The point cloud removes every ambiguous knob (`D`, `f_fill`, the classifier thresholds); they
survive only on the fallback path.
