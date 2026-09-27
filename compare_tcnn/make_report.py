"""Turn bench_tcnn output (results/speed.csv, results/conv.csv) into results/report.md + charts.

    python compare_tcnn/make_report.py [--meta "free text line"]

Ratio convention everywhere: tcnn_ms / tinymlp_ms  (> 1 means TinyMLP is faster).
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

MODELS = [("hashgrid", "Hash grid + MLP (16 outputs)"),
          ("mlp", "Plain MLP (in = out = hidden)"),
          ("nerf_color", "3DMaker colour head (32 -> 32 -> 32 -> 3, sigmoid)")]
OPS = ["inference", "backward", "train"]
RIVALS = ["tcnn", "tcnn_jit"]


def load_speed(path):
    t = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["op"], int(r["hidden"]), int(r["layers"]), int(r["batch"]))
            t.setdefault(key, {})[r["framework"]] = float(r["ms"])
    return t


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def fmt_ratio(x):
    if x != x:
        return "—"
    return f"**{x:.2f}×**" if x >= 1 else f"{x:.2f}×"


def bsz(b):
    return f"{b // 1024}K" if b < 1 << 20 else f"{b >> 20}M"


def speed_section(t, out, figs):
    configs = sorted({(k[2], k[3], k[4]) for k in t})
    batches = sorted({k[4] for k in t})

    out.append("## Headline — geometric-mean ratio (tcnn ms ÷ TinyMLP ms, >1 = TinyMLP faster)\n")
    out.append("| model | op | vs tcnn (AOT fully fused) | vs tcnn JIT |")
    out.append("|---|---|---|---|")
    for m, _ in MODELS:
        for op in OPS:
            cells = []
            for rv in RIVALS:
                rs = [v[rv] / v["tinymlp"] for k, v in t.items()
                      if k[0] == m and k[1] == op and rv in v and "tinymlp" in v]
                cells.append(fmt_ratio(geomean(rs)) + (f" <sub>(n={len(rs)})</sub>" if rs else ""))
            out.append(f"| {m} | {op} | {cells[0]} | {cells[1]} |")
    out.append("")

    # NeRF-relevant spotlight: what 3DMaker actually runs per training step.
    out.append("## What 3DMaker actually runs (batch 256K)\n")
    out.append("Density head = hash grid, hidden 64, 2 layers; colour head = `nerf_color`.\n")
    out.append("| model | op | TinyMLP ms | tcnn ms | tcnn JIT ms | vs tcnn | vs JIT |")
    out.append("|---|---|---|---|---|---|---|")
    for m, H, L in [("hashgrid", 64, 2), ("nerf_color", 32, 3)]:
        for op in OPS:
            v = t.get((m, op, H, L, 1 << 18), {})
            if not v:
                continue
            tm, tc, tj = v.get("tinymlp"), v.get("tcnn"), v.get("tcnn_jit")
            f = lambda x: f"{x:.3f}" if x else "—"
            r1 = tc / tm if tm and tc else float("nan")
            r2 = tj / tm if tm and tj else float("nan")
            out.append(f"| {m} | {op} | {f(tm)} | {f(tc)} | {f(tj)} | {fmt_ratio(r1)} | {fmt_ratio(r2)} |")
    out.append("")

    # Chart: ratio at the largest batch, per model/op/config.
    big = max(batches)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, op in zip(axes, OPS):
        labels, a, j = [], [], []
        for m, _ in MODELS:
            for H, L, B in configs:
                v = t.get((m, op, H, L, B))
                if B != big or not v or "tinymlp" not in v:
                    continue
                labels.append(f"{m[:4]} H{H}/L{L}")
                a.append(v["tcnn"] / v["tinymlp"] if "tcnn" in v else 0)
                j.append(v["tcnn_jit"] / v["tinymlp"] if "tcnn_jit" in v else 0)
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], a, 0.4, label="vs tcnn")
        ax.bar([i + 0.2 for i in x], j, 0.4, label="vs tcnn JIT")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=70, fontsize=7)
        ax.set_title(f"{op} @ batch {bsz(big)}")
    axes[0].set_ylabel("tcnn ms / TinyMLP ms  (>1 = TinyMLP faster)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_ratio.png"), dpi=130)
    figs.append(("Speed ratio at the largest batch", "fig_ratio.png"))

    out.append("## Full results (ms — TinyMLP / tcnn / tcnn JIT)\n")
    for m, title in MODELS:
        out.append(f"### {title}\n")
        for op in OPS:
            rows = sorted({(k[2], k[3]) for k in t if k[0] == m and k[1] == op})
            if not rows:
                continue
            out.append(f"**{op}**\n")
            out.append("| H/L \\ batch | " + " | ".join(bsz(b) for b in batches) + " |")
            out.append("|---" * (len(batches) + 1) + "|")
            for H, L in rows:
                cells = []
                for B in batches:
                    v = t.get((m, op, H, L, B), {})
                    g = lambda k: f"{v[k]:.3f}" if k in v else "—"
                    cells.append(f"{g('tinymlp')} / {g('tcnn')} / {g('tcnn_jit')}")
                out.append(f"| {H}/{L} | " + " | ".join(cells) + " |")
            out.append("")


def conv_section(path, out, figs):
    runs = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            runs[r["framework"]].append((int(r["step"]), float(r["test_mse"]), float(r["train_ms"])))
    if not runs:
        return
    steps = sorted({s for v in runs.values() for s, _, _ in v})
    fws = [fw for fw in ["tinymlp", "tcnn", "tcnn_jit"] if fw in runs]
    out.append("## Convergence — same hash-grid model, same data, same Adam settings\n")
    out.append("Hidden 64 / 2 layers, batch 256K, 16-channel sinusoid field (1–31 cycles per unit), "
               "Adam lr 1e-2, β=(0.9, 0.99), ε=1e-15. Test MSE on 256K held-out points "
               "(lower is better); `train ms` is cumulative GPU time spent in training steps.\n")
    out.append("| step | " + " | ".join(f"{fw} MSE" for fw in fws) + " | "
               + " | ".join(f"{fw} train ms" for fw in fws) + " |")
    out.append("|---" * (1 + 2 * len(fws)) + "|")
    for s in steps:
        d = {fw: {st: (m, ms) for st, m, ms in runs[fw]} for fw in fws}
        mse = [f"{d[fw][s][0]:.5f}" if s in d[fw] else "—" for fw in fws]
        tms = [f"{d[fw][s][1]:.0f}" if s in d[fw] else "—" for fw in fws]
        out.append(f"| {s} | " + " | ".join(mse) + " | " + " | ".join(tms) + " |")
    out.append("")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for fw in fws:
        v = [x for x in runs[fw] if x[0] > 0]
        a1.plot([x[0] for x in v], [x[1] for x in v], marker="o", ms=3, label=fw)
        a2.plot([x[2] / 1000 for x in v], [x[1] for x in v], marker="o", ms=3, label=fw)
    for a, xl in [(a1, "training step"), (a2, "cumulative training GPU time (s)")]:
        a.set_yscale("log"); a.set_xlabel(xl); a.set_ylabel("test MSE"); a.grid(alpha=0.3); a.legend()
    a1.set_title("Quality per step"); a2.set_title("Quality per second")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_convergence.png"), dpi=130)
    figs.append(("Convergence", "fig_convergence.png"))


def ablate_section(path, out):
    rows = defaultdict(dict)  # (target, framework) -> {step: mse}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(r["target"], r["framework"])][int(r["step"])] = float(r["test_mse"])
    if not rows:
        return
    targets = sorted({t for t, _ in rows}, key=lambda t: t != "offset")
    fws = ["tinymlp", "tinymlp_nobias", "tinymlp_nobias_tcnninit", "tcnn"]
    names = {"tinymlp": "TinyMLP (biases, own init)", "tinymlp_nobias": "TinyMLP, biases frozen at 0",
             "tinymlp_nobias_tcnninit": "TinyMLP, no biases + tcnn init", "tcnn": "tcnn"}
    out.append("## Ablation — why does TinyMLP reach lower error?\n")
    out.append("Same convergence setup. `offset` is the field above (0.5 + 0.4·sin, so a constant "
               "output bias is free accuracy); `zeromean` is 0.4·sin with no DC term. Test MSE:\n")
    out.append("| variant | " + " | ".join(f"{t} @100 | {t} @1000 | {t} @3000" for t in targets) + " |")
    out.append("|---" * (1 + 3 * len(targets)) + "|")
    for fw in fws:
        cells = []
        for t in targets:
            d = rows.get((t, fw), {})
            cells += [f"{d[s]:.5f}" if s in d else "—" for s in (100, 1000, 3000)]
        out.append(f"| {names[fw]} | " + " | ".join(cells) + " |")
    out.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="append", default=[])
    args = ap.parse_args()

    out = ["# TinyMLP vs tiny-cuda-nn\n"]
    out += [f"- {m}" for m in args.meta]
    out.append("- Both libraries built by the same nvcc for the same arch with `--use_fast_math`, linked "
               "into one executable, timed with identical CUDA-event code on the same stream; each number "
               "is the median of 3 runs of N back-to-back launches.")
    out.append("- `tcnn` = precompiled FullyFusedMLP (+ HashGrid); `tcnn JIT` = tiny-cuda-nn 2.x runtime "
               "fusion (`set_jit_fusion(true)`), which fuses encoding + MLP (and the whole train step) into "
               "one NVRTC kernel. JIT has no standalone backward, so that column is blank.")
    out.append("- Parity caveats: tcnn's MLP has **no biases** (TinyMLP has them); loss scale is each "
               "library's default (tcnn 128, TinyMLP 65536); outputs are fp32 on both sides; the "
               "convergence/ablation runs use 3-float positions (affects speed only, not quality).\n")
    figs = []
    sp = os.path.join(RES, "speed.csv")
    if os.path.exists(sp):
        speed_section(load_speed(sp), out, figs)
    cp = os.path.join(RES, "conv.csv")
    if os.path.exists(cp):
        conv_section(cp, out, figs)
    ap_ = os.path.join(RES, "ablate.csv")
    if os.path.exists(ap_):
        ablate_section(ap_, out)
    out.append("## Charts\n")
    out += [f"**{t}**\n\n![{t}]({p})\n" for t, p in figs]
    with open(os.path.join(RES, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("wrote", os.path.join(RES, "report.md"))


if __name__ == "__main__":
    main()
