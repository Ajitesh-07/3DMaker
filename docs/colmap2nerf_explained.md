Good direction. Before any code, let me work out what's actually recoverable from transforms.json, because that determines what the algorithm can and can't honestly compute — and there's a real asymmetry between the AABB (recoverable) and num_cascades (only partly).

What signals the file actually gives you

From each frame's c2w (4×4) you can extract, in your engine's OpenGL convention:

- Camera center Cᵢ = (c[0][3], c[1][3], c[2][3]) — translation column.
- Forward (view) direction fᵢ = −(c[0][2], c[1][2], c[2][2]) — the camera looks down −Z, matching your generate_custom_rays_kernel (dir_z = −1).
- Up (c[0][1], c[1][1], c[2][1]).
- Plus the global camera_angle_x (FOV).

What you do not have: any 3D geometry. No point cloud, no depth. This is the crux, and it splits the problem cleanly:

┌─────────────────────┬────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────┐
│      Quantity       │      Recoverable from poses?       │                                      Why                                      │
├─────────────────────┼────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ Scene center        │ ✅ Yes, robustly                   │ rays converge on what was photographed                                        │
├─────────────────────┼────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ Object scale / AABB │ ⚠️ Yes, up to a framing assumption │ poses have a scale ambiguity; FOV + "subject fills frame" breaks it           │
├─────────────────────┼────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ num_cascades        │ ❌ Not reliably                    │ cascades measure how far content extends; cameras being far ≠ scene being big │
└─────────────────────┴────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────┘

Let me derive each.

Step 1 — Scene center via least-squares ray convergence

The object is wherever all the cameras are looking. Each camera defines a ray (Cᵢ, fᵢ). Find the point p minimizing the sum of squared perpendicular distances to all those rays.

The perpendicular component of (p − Cᵢ) relative to direction fᵢ is (p − Cᵢ) − ((p − Cᵢ)·fᵢ) fᵢ. Writing the projector onto the plane ⟂ fᵢ as

$$A_i = I - f_i f_i^{\top}$$

the cost is $\sum_i (p - C_i)^{\top} A_i (p - C_i)$. Setting the gradient to zero gives a single 3×3 linear system:

$$\Big(\underbrace{\sum_i A_i}{M}\Big), p = \underbrace{\sum_i A_i C_i}{b}, \qquad p = M^{-1} b$$

M = 0 (3x3); b = 0 (3)
for each frame:
    f = normalize(-col2);  C = col3
    A = I - f·fᵀ
    M += A;  b += A·C
center = solve(M, b)        // 3x3 solve (Cramer / Gauss)

Degeneracy guard: if all cameras look the same way (forward-facing capture), the rays are parallel, M is rank-deficient, and the solve blows up. Detect via det(M) (or condition number) below a threshold → fall back to the centroid of camera centers and flag the scene as "forward-facing / unbounded." Optionally add λI (small Tikhonov term) to M for numerical safety.

This is exactly the central_point trick instant-ngp uses, and it's robust for any orbit/turntable/inward-facing capture.

Step 2 — Object scale (breaking the pose scale ambiguity)

Poses alone can't tell a small object shot up close from a large one shot from afar — uniform scaling of the whole scene is unobservable. You need one extra assumption. Two independent estimators; use both and cross-check:

(a) Framing estimator (uses FOV — the good one). Photographers frame the subject to roughly fill the frame. So the subject's angular radius ≈ a fixed fraction of the FOV. At camera distance dᵢ = ‖Cᵢ − center‖:

$$r_{\text{obj}} \approx \operatorname{median}i(d_i)\cdot \tan!\Big(\tfrac{\text{camera_angle_x}}{2}\Big)\cdot f{\text{fill}}, \qquad f_{\text{fill}} \approx 0.5$$

This genuinely uses information poses+intrinsics contain. f_fill ≈ 0.5 says "subject spans about half the frame"; tune per dataset.

(b) Distance-shell estimator (sanity bound). The subject must sit in front of the nearest camera, so r_obj ≤ min_i dᵢ. Use a robust low percentile (p10 of dᵢ) rather than the raw min to reject one stray close pose.

Take r_obj = clamp(framing_estimate, small, p10(dᵢ)). The framing estimate is your primary; the shell is a ceiling.

Then normalize (mirroring what colmap2nerf.py does, but pose-only): scale = 1 / r_obj, shift by −center, and the object lands in the unit sphere — so your existing ±1.5 base box fits with the same headroom we discussed earlier. Important integration note: if colmap2nerf already normalized the scene, doing it again here double-normalizes. So pick one owner (see Step 5).

Step 3 — Build the AABB

Once centered and scaled so r_obj ≈ 1:

base = 1.5                       // must match opts.aabbMax today
aabbMin = center - base          // (after scaling, center≈0 → ±1.5)
aabbMax = center + base

Refinements worth having:
- Per-axis box. For a turntable, the vertical extent of real content is often much smaller than horizontal. Instead of a cube, take robust percentiles of camera positions per axis and pad — gives an anisotropic box that excludes the empty upper volume (directly attacks the fog problem from the last question).
- Frustum-intersection refinement (optional, stronger). The subject lies in the intersection of all camera frustums (its visual hull bound). Intersecting the frustums and taking the bounding box of that region gives a tighter, geometry-aware AABB than camera distances alone — at the cost of more code.

Step 4 — num_cascades: the honest part

Cascades exist to cover content that extends far beyond the subject (background). The formula you already have,

$$\texttt{num_cascades} = \operatorname{clamp}\Big(\big\lceil \log_2(\text{ratio}/1.5)\big\rceil + 1,; 1,; 6\Big), \quad \text{ratio} = \frac{\text{scene_radius}}{\text{obj_radius}},$$

needs scene_radius = how far geometry reaches. Camera distance is not a valid proxy for this — in a turntable, cameras sit at 3–4× the object radius but the scene radius equals the object radius, so ratio ≈ 1 → 1 cascade. If you plugged camera distance in, you'd compute ratio ≈ 4 and wrongly demand 3 cascades for a bounded object. This is why colmap2nerf uses point distances, not camera distances.

So from poses alone, the principled answer is:

- Default num_cascades = 1 (assume bounded/object-centric), and
- Bump only on a geometric signal poses can expose: the convergence quality. If the ray-convergence residual is small and cameras form a tight inward-looking ring, it's almost certainly a bounded object → 1 cascade. If M was near-singular / cameras face outward or scatter (panorama, forward-facing drive), the scene is unbounded → raise cascades (e.g. 4–6). Concretely, classify by mean ray residual and the spread of forward directions:

spread = variance of forward dirs fᵢ
residual = mean perpendicular distance of center to rays / r_obj
if residual small AND cameras inward-looking:   num_cascades = 1   // bounded
elif forward-facing / outward / high spread:     num_cascades = 4..6 // unbounded
else:                                            num_cascades = 2

This is a coarse classifier, not the precise point-cloud computation — and the algorithm should say so (log it), not pretend it measured the background extent.

Recommendation: for a trustworthy num_cascades, the DataLoader should consume geometry when it exists — read points3D.txt (or a sidecar in the json) and run the existing colmap2nerf cascade math engine-side, falling back to the pose-only classifier above only when no points are available. That keeps the good estimate where geometry exists and degrades gracefully otherwise.

Step 5 — Where this slots in (avoid double work)

You currently have: colmap2nerf.py bakes center+scale into poses and writes num_cascades; Pipeline.cu hardcodes ±1.5; DataLoader reads num_cascades. To move derivation into the DataLoader, pick an ownership model:

- Option A — DataLoader owns normalization. colmap2nerf writes raw (un-normalized) poses + point cloud; DataLoader runs Steps 1–4 at load time and exposes getAABB(), getCenter(), getNumCascades(); Pipeline.cu reads opts.aabbMin/Max from the DataLoader instead of hardcoding. Cleanest, works on any transforms.json (including external NeRF datasets with no num_cascades).
- Option B — Keep colmap2nerf normalization, add a pose-only AABB tightening in the DataLoader. Less invasive; DataLoader assumes poses are already ~unit-normalized (so center ≈ 0, r_obj ≈ 1) and only computes a tighter per-axis AABB + the cascade classifier. Good if you don't want to touch the Python path.

Either way Pipeline.cu:111-112 stops being a hardcoded constant and reads from the loader, and getNumCascades() becomes geometry-aware.

Summary of the algorithm

1. centers Cᵢ, forwards fᵢ  ← from c2w columns
2. center  ← solve (Σ Aᵢ) p = Σ Aᵢ Cᵢ,  Aᵢ = I − fᵢfᵢᵀ   (centroid fallback if singular)
3. r_obj   ← median(dᵢ)·tan(FOV/2)·f_fill,  capped at p10(dᵢ)
4. scale=1/r_obj, recenter  →  object in unit sphere
5. AABB    ← center ± 1.5  (or per-axis percentile box)
6. num_cascades ← point-cloud math if points available, else convergence-based classifier (default 1)

The one thing I'd push back on: don't try to compute num_cascades precisely from poses — it's geometrically unidentifiable. Compute center and AABB from poses (those are sound), and either read geometry for cascades or fall back to a clearly-labeled heuristic.

Want me to implement this in the DataLoader — and if so, Option A or B? I'd also need to know whether you're willing to have colmap2nerf emit the point cloud (or raw poses) so the loader can do the geometry-accurate cascade count.