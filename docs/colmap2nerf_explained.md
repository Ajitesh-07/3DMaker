Yeah, let me build it from scratch — it's not arbitrary, it falls out of one physical idea.

The core question: what's a "fair" unit of distance?

Raw t (meters) is the obvious coordinate, but it's the wrong one for vision, because equal steps in meters are not equal in visual importance.

Walk a camera through a scene:
- Moving from t=1 to t=2 (one meter) — close objects swing across the frame. Huge change.
- Moving from t=100 to t=101 (also one meter) — the far hill barely shifts a pixel. Almost nothing.
- Moving from t=100 all the way to t=∞ — the entire infinite remaining distance — produces about as much visual change as that first single meter did.

So the thing that scales with "visual / representational distance" is not t — it's 1/t. This is exactly why stereo vision works in disparity (= 1/depth): equal disparity steps carry equal information. A z-buffer in graphics is nonlinear in 1/z for the same reason. So if we want a coordinate where "equal steps = equally important," we should build it out of 1/t, not t.

Building g(t), checking it piece by piece

Start with the raw idea: use 1/t. But raw 1/t runs from 1/t_n (at the near plane) down to 0 (at infinity) — it decreases with distance, and it's not normalized. We want something that increases from near to far and lands in a clean range. So define:

$$g(t) = 1 - \frac{t_n}{t}$$

Check the endpoints — that's all this formula is, the two anchors:

- At the near plane t = t_n:  g = 1 − t_n/t_n = 1 − 1 = 0. ✓ near → 0
- At infinity t = ∞:  g = 1 − 0 = 1. ✓ far → 1
- Halfway-ish t = 2·t_n:  g = 1 − ½ = 0.5.

That's the whole construction. The 1 − (…) flips it so it increases; the t_n in the numerator is just the scale that pins the near plane to exactly 0. The far field compresses for free, because it's 1/t underneath.

The outer normalization is just "pin the endpoints"

g(t) already lands in [0,1) if the far plane is literally infinity. But in practice your far plane t_f is finite (the marching reach / contraction boundary), so g(t_f) is a bit less than 1. To make s hit exactly 1 at your actual far plane, you affine-rescale:

$$s(t) = \frac{g(t) - g(t_n)}{g(t_f) - g(t_n)}$$

Since g(t_n) = 0, this is just s(t) = g(t) / g(t_f) — "stretch g so it reaches 1 at t_f instead of at ∞." The general form with both t_n and t_f is nothing but the standard affine map that forces s(t_n)=0 and s(t_f)=1. If t_f=∞, the denominator is 1 and s = g exactly.

Watch the compression happen

Take t_n = 1, far plane at ∞ (so s = 1 − 1/t):

┌────────────┬──────┐
│ t (meters) │  s   │
├────────────┼──────┤
│          1 │ 0.00 │
├────────────┼──────┤
│          2 │ 0.50 │
├────────────┼──────┤
│          4 │ 0.75 │
├────────────┼──────┤
│         10 │ 0.90 │
├────────────┼──────┤
│        100 │ 0.99 │
├────────────┼──────┤
│          ∞ │ 1.00 │
└────────────┴──────┘

Look at the distribution of the s-budget:
- The first meter [1,2] of foreground eats half of all of s.
- The entire background [10, ∞) is crammed into the last 0.1 of s.

That's the payoff. In s-space the foreground gets most of the resolution (and most of the regularization budget), and the infinite background is packed into a sliver — so background samples have |s_i − s_j| ≈ 0.002 instead of |t_i − t_j| = 5. The distortion loss stops seeing the background as "spread out," stops crushing it into a shell, and lets it keep a smooth depth.

Why this is the right space, not just a compressing space

The clincher: your engine already warps space this exact way. contract_pos maps radius r > 1 to 2 − 1/r — that's also "1 minus something over distance," approaching a finite boundary (radius 2) as r→∞. Same family as 1 − t_n/t. So measuring distortion in s is measuring it in (approximately) the same compressed space the MLP actually represents the scene in. The disparity warp and your contraction are the same 1/distance idea with different constants — which is why the disparity s captures nearly all the benefit even though it's not bit-for-bit your contraction profile.

The one-line summary: s is just t measured in disparity (1/depth) instead of meters, rescaled to [0,1] — and disparity is the coordinate where equal steps carry equal visual weight, which is exactly the fairness the distortion loss needs to treat foreground and background even-handedly.