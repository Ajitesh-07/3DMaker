# Distortion Loss: from $\partial\mathcal{L}/\partial w$ to $\partial\mathcal{L}/\partial\sigma$

This note derives, in full, how the mip‑NeRF‑360 distortion loss gradient flows
from the **weights** $w_i$ back to the **density** $\sigma_i$ that the MLP actually
produces. The one conceptual hurdle — *why $g_k$ appears inside the suffix sum* —
is isolated in §4.

---

## 1. Setup and notation

Along a single ray we have samples $i = 0, 1, \dots, N-1$, sorted front‑to‑back.

| symbol | meaning |
|---|---|
| $\delta_i$ | length of interval $i$ (`delta_t`) |
| $\sigma_i$ | density at sample $i$ (output of the density MLP, after $e^{z_i-b}$) |
| $\alpha_i = 1 - e^{-\sigma_i\delta_i}$ | opacity of interval $i$ |
| $T_i = \prod_{j<i}(1-\alpha_j)$ | transmittance reaching sample $i$ (with $T_0 = 1$) |
| $w_i = T_i\,\alpha_i$ | volume‑rendering weight |
| $m_i$ | interval midpoint, $m_i = t_i + \delta_i/2$ |

The dependency chain we must differentiate through is

$$
\sigma_i \;\longrightarrow\; \big(\alpha_i,\; T_{k>i}\big) \;\longrightarrow\; w \;\longrightarrow\; \mathcal{L}.
$$

The middle step is the tricky one: $\sigma_i$ (via $\alpha_i$) feeds **not just $w_i$**
but every weight behind it, because $\alpha_i$ sits inside $T_k$ for all $k>i$.

---

## 2. The loss and $\partial\mathcal{L}/\partial w$ (recap)

$$
\mathcal{L} \;=\; \lambda\!\left(
\underbrace{\sum_{i,j} w_i\,w_j\,\lvert m_i - m_j\rvert}_{\mathcal{L}_{\mathrm{bi}}}
\;+\;
\underbrace{\tfrac{1}{3}\sum_i w_i^{2}\,\delta_i}_{\mathcal{L}_{\mathrm{self}}}
\right).
$$

We already compute its derivative with respect to each weight in the render kernel
(this is `dw_out[i]`):

$$
\boxed{\,g_i \;\equiv\; \frac{\partial\mathcal{L}}{\partial w_i}
\;=\; \lambda\!\left(2\sum_{j} w_j\,\lvert m_i - m_j\rvert \;+\; \tfrac{2}{3}\,w_i\,\delta_i\right).}
$$

From here on $g_i$ is a **known number per sample** — it is exactly what the
distortion kernel writes into `dw_out`. The rest of this note never re‑opens it;
we only ask how a change in $\sigma_i$ changes $\mathcal{L}$ *given* these $g$'s.

---

## 3. The gradient we need

We want $\partial\mathcal{L}/\partial\sigma_i$. We get there in two hops:

$$
\frac{\partial\mathcal{L}}{\partial\sigma_i}
= \frac{\partial\mathcal{L}}{\partial\alpha_i}\cdot\frac{\partial\alpha_i}{\partial\sigma_i}.
$$

§4–§6 compute $\partial\mathcal{L}/\partial\alpha_i$; §7 does the second factor.

---

## 4. Why $g_k$ lives *inside* the suffix sum  ⭐

This is the crux. The tempting (wrong) move is to treat $w_i$ as the only weight
that $\alpha_i$ touches:

$$
\frac{\partial\mathcal{L}}{\partial\alpha_i}
\;\stackrel{?}{=}\; g_i\,\frac{\partial w_i}{\partial\alpha_i}
\qquad\textbf{(WRONG)}.
$$

That would be correct only if $w_i$ were the *single* output depending on
$\alpha_i$. It isn't: $\alpha_i$ is one scalar that feeds into **many** weights
($w_i$ and every $w_k$ with $k>i$, through $T_k$). The multivariate chain rule
says: sum over **all** outputs that depend on $\alpha_i$, each weighted by the
loss's sensitivity to *that* output:

$$
\boxed{\;\frac{\partial\mathcal{L}}{\partial\alpha_i}
= \sum_{k}\frac{\partial\mathcal{L}}{\partial w_k}\,\frac{\partial w_k}{\partial\alpha_i}
= \sum_{k} g_k\,\frac{\partial w_k}{\partial\alpha_i}.\;}
$$

Each term carries its **own** $g_k = \partial\mathcal{L}/\partial w_k$ — *not* a
global $g_i$. That is the whole answer to "how did the $g_k$ get inside the sum?":
it was never $g_i$ times one derivative; it is a **sum of products** $g_k\cdot(\partial w_k/\partial\alpha_i)$.

### Concrete 3‑sample example

Take $N=3$, and differentiate w.r.t. $\alpha_1$. The weights are
$w_0 = T_0\alpha_0$, $w_1 = T_1\alpha_1$, $w_2 = T_2\alpha_2$ with
$T_2 = (1-\alpha_0)(1-\alpha_1)$. Which weights contain $\alpha_1$?

- $w_0$: no $\alpha_1$ → contributes $0$.
- $w_1 = T_1\alpha_1$: yes, directly → $\partial w_1/\partial\alpha_1 = T_1$.
- $w_2 = T_2\alpha_2$: yes, through $T_2$ → $\partial w_2/\partial\alpha_1 = -w_2/(1-\alpha_1)$.

So

$$
\frac{\partial\mathcal{L}}{\partial\alpha_1}
= g_1\,T_1 \;+\; g_2\left(-\frac{w_2}{1-\alpha_1}\right).
$$

The "suffix" for $i=1$ is the single term $g_2 w_2$ — sample $2$'s **own** $g_2$.
With more samples it becomes $\sum_{k>i} g_k w_k$: every later sample contributes
*its* $g_k$. There is no way to pull a common $g_i$ out front.

---

## 5. The per‑weight derivatives $\partial w_k/\partial\alpha_i$

Generalising the example:

$$
\frac{\partial w_k}{\partial\alpha_i}=
\begin{cases}
0, & k < i \quad(\alpha_i \text{ not in } T_k \text{ or } \alpha_k)\\[4pt]
T_i, & k = i \quad(w_i = T_i\alpha_i,\ T_i \text{ independent of } \alpha_i)\\[4pt]
-\dfrac{w_k}{1-\alpha_i}, & k > i \quad\left(T_k \ni (1-\alpha_i),\ \partial T_k/\partial\alpha_i = -T_k/(1-\alpha_i)\right)
\end{cases}
$$

For the $k>i$ case explicitly: $T_k = \prod_{j<k}(1-\alpha_j)$ contains the factor
$(1-\alpha_i)$, so

$$
\frac{\partial T_k}{\partial\alpha_i}
= -\!\!\prod_{\substack{j<k\\ j\neq i}}(1-\alpha_j)
= -\frac{T_k}{1-\alpha_i},
\qquad
\frac{\partial w_k}{\partial\alpha_i}
= \alpha_k\frac{\partial T_k}{\partial\alpha_i}
= -\frac{\alpha_k T_k}{1-\alpha_i}
= -\frac{w_k}{1-\alpha_i}.
$$

---

## 6. Result: $\partial\mathcal{L}/\partial\alpha_i$

Substitute §5 into the chain‑rule sum from §4 (the $k<i$ terms vanish):

$$
\boxed{\;\frac{\partial\mathcal{L}}{\partial\alpha_i}
= g_i\,T_i \;-\; \frac{1}{1-\alpha_i}\sum_{k>i} g_k\,w_k.\;}
$$

Two things to read off, matching the corrections discussed:

1. The self term is $g_i\,T_i$ — it **does** carry $g_i$ (easy to drop).
2. The suffix term has a **minus** sign, and it is $\sum_{k>i} g_k w_k$
   (each weight times its own $g_k$), **not** $\sum_{k>i} w_k$.

Physically: pushing more opacity into sample $i$ steals transmittance from
everything behind it, shrinking those weights — so the suffix pulls the gradient
down, scaled by how much the loss cared about each of those weights.

---

## 7. Chaining to $\sigma$: the $(1-\alpha)$ cancellation

The second factor:

$$
\alpha_i = 1 - e^{-\sigma_i\delta_i}
\quad\Longrightarrow\quad
\frac{\partial\alpha_i}{\partial\sigma_i}
= \delta_i\,e^{-\sigma_i\delta_i}
= \delta_i\,(1-\alpha_i).
$$

Therefore

$$
\frac{\partial\mathcal{L}}{\partial\sigma_i}
= \delta_i(1-\alpha_i)\!\left[\,g_i T_i - \frac{1}{1-\alpha_i}\sum_{k>i} g_k w_k\right]
= \delta_i\!\left[\,g_i\,T_i\,(1-\alpha_i) - \sum_{k>i} g_k w_k\right].
$$

$$
\boxed{\;\frac{\partial\mathcal{L}}{\partial\sigma_i}
= \delta_i\!\left[\,g_i\,T_i\,(1-\alpha_i) \;-\; \sum_{k>i} g_k\,w_k\right].\;}
$$

The $(1-\alpha_i)$ from $\partial\alpha/\partial\sigma$ **cancels** the
$1/(1-\alpha_i)$ in the suffix. This is why we implement *this* form directly and
never materialise $\partial\mathcal{L}/\partial\alpha_i$: the explicit
$1/(1-\alpha_i)$ blows up for a near‑opaque sample ($\alpha_i\to 1$), even though
it cancels analytically.

---

## 8. From $\sigma$ to the density‑logit gradient

The MLP emits a logit $z_i$; density is $\sigma_i = e^{\,z_i - b}$
(`expf(fminf(logit - densityBias, 8))`), so $\partial\sigma_i/\partial z_i = \sigma_i$.
Hence

$$
\frac{\partial\mathcal{L}}{\partial z_i}
= \sigma_i\,\frac{\partial\mathcal{L}}{\partial\sigma_i},
$$

and finally we apply the global `loss_scale` (fp16 gradient scaling) and
`batch_scale` $= 1/N_{\text{rays}}$ that every gradient in this codebase carries:

$$
\texttt{ds}_i = \texttt{loss\_scale}\cdot\texttt{batch\_scale}\cdot\sigma_i\cdot
\delta_i\!\left[g_i T_i(1-\alpha_i) - \sum_{k>i} g_k w_k\right].
$$

This is the value added into `tmp_dsigma[i]` (on top of the photometric term).

---

## 9. Linear‑scan implementation (prefix / suffix)

The only non‑local piece is $\sum_{k>i} g_k w_k$. Define the per‑ray total and
the running prefix:

$$
G \;=\; \sum_{k} g_k w_k,
\qquad
G^{\le i} \;=\; \sum_{k\le i} g_k w_k,
\qquad
\sum_{k>i} g_k w_k \;=\; G - G^{\le i}.
$$

So the backward is a single forward scan once $G$ is known:

- **Render kernel** (it already knows every $w_k$ and computes every $g_k$):
  accumulate $G = \sum_k g_k w_k$ and store it per ray
  (`weight_sum[r] = G`). *(Note: this is $\sum g_k w_k$, **not** $\sum w_k$.)*
- **`compute_color_grad`** scan, at sample $i$ (it reconstructs $T_i,\alpha_i,w_i$):

```
g_i      = dw_out[i]
G_pre   += w_i * g_i               // now includes i  ->  suffix = G - G_pre = sum_{k>i}
g_suff   = G - G_pre
d_sigma_dist = g_i * T_i * (1 - alpha_i) - g_suff
d_sigma_i = delta_i * ( (photometric bracket) + d_sigma_dist )
ds        = loss_scale * batch_scale * sigma_i * d_sigma_i
```

This mirrors the existing **photometric** σ‑gradient one‑for‑one:

| photometric | distortion |
|---|---|
| per‑sample scalar $P_k = \sum_{c}\phi_c\,c_{c,k}$ | $g_k = $ `dw_out[k]` |
| total $\sum_k w_k P_k$ via `final_rgb` | total $G=\sum_k w_k g_k$ via `weight_sum` |
| running prefix via `currentRgb` | running prefix `G_pre` |
| suffix `final_rgb − currentRgb` | suffix `G − G_pre` |

The photometric path has *always* had this exact two‑sided ($k=i$ self + $k>i$
suffix) structure — see `ds_* = currentT*c_*(1-alpha) - suff_*` in
`compute_color_grad`. Distortion is the same shape with $c_k \to g_k$.

---

## 10. Mapping to the code

- **`compositing.cu` / `render_rays_distortion_kernel`** — pass 2 finalises
  `dw_out[i] = g_i` and accumulates `G = Σ g_k w_k`, written to `weight_sum[r]`.
- **`otherKernels.cu` / `compute_color_grad`** — reads `g_i = dw_out[i]` and
  `G = weight_sum[r]`, tracks `G_pre`, and adds
  $\delta_i\,[\,g_i T_i(1-\alpha_i) - (G - G^{\le i})\,]$ into `d_sigma_i` before
  the shared $\times\,\sigma_i\times$`loss_scale`$\times$`batch_scale`.
- **Toggle** — because $g_k \propto \lambda$, setting $\lambda = 0$ makes every
  $g_k = 0$ and the whole distortion contribution vanishes cleanly. A/B is just
  `lambdaDist`.

### Common mistakes (all corrected above)

1. Using $g_i$ as a global factor instead of summing $g_k w_k$ inside the suffix (§4).
2. Forgetting the $g_i$ on the self term (writing $T_i$ instead of $g_i T_i$).
3. Sign of the suffix: it is **minus** (§6).
4. Storing $\sum w_k$ instead of $\sum g_k w_k$ in `weight_sum`.
5. Materialising $1/(1-\alpha_i)$ explicitly — unstable; use the cancelled
   $\sigma$‑form from §7.
