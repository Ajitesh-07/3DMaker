# Combine tinymlp.csv + pytorch.csv -> charts (PNG) + report.md
import csv, os, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
OPS = ['inference', 'backward', 'train']; MODELS = ['mlp', 'hashgrid']
HIDDENS = [32, 64, 128]; LAYERS = [2, 4]; BATCHES = [65536, 262144, 1048576]

def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['ms'] == 'nan': continue
            d[(r['model'], r['op'], int(r['hidden']), int(r['layers']), int(r['batch']))] = float(r['ms'])
    return d

tm = load(os.path.join(RES, 'tinymlp.csv'))
pt = load(os.path.join(RES, 'pytorch.csv'))
def sp(k): return pt[k]/tm[k] if (k in pt and k in tm and tm[k] > 0) else float('nan')
def geomean(v):
    v = [x for x in v if x == x and x > 0]
    return math.exp(sum(math.log(x) for x in v)/len(v)) if v else float('nan')

# ---------- Chart 1/2: latency vs batch per model (H=64, L=4) ----------
for model in MODELS:
    H, L = 64, 4
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, op in zip(axs, OPS):
        tv = [tm.get((model, op, H, L, B), np.nan) for B in BATCHES]
        pv = [pt.get((model, op, H, L, B), np.nan) for B in BATCHES]
        ax.plot(BATCHES, tv, 'o-', color='#1f77b4', lw=2.2, ms=7, label='TinyMLP')
        ax.plot(BATCHES, pv, 's--', color='#d62728', lw=2.2, ms=7, label='PyTorch (fp16)')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_title(op, fontweight='bold'); ax.set_xlabel('batch size'); ax.set_ylabel('latency (ms)')
        ax.grid(True, which='both', alpha=0.25); ax.legend()
        for B, t, p in zip(BATCHES, tv, pv):
            if t == t and p == p: ax.annotate(f'{p/t:.1f}x', (B, math.sqrt(t*p)), fontsize=8, ha='center', color='#444')
    fig.suptitle(f'{model.upper()} latency — TinyMLP vs PyTorch  (hidden={H}, layers={L})  [annotated = speedup]', fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(RES, f'fig_{model}_latency.png'), dpi=110); plt.close(fig)

# ---------- Chart 3: speedup bars per op/model at 1M, H=64, L=4 ----------
fig, ax = plt.subplots(figsize=(9, 5.2)); H, L, B = 64, 4, 1048576; x = np.arange(len(OPS)); w = 0.36
for i, model in enumerate(MODELS):
    vals = [sp((model, op, H, L, B)) for op in OPS]
    bars = ax.bar(x + (i-0.5)*w, vals, w, label=model.upper(), color=['#1f77b4', '#ff7f0e'][i])
    for b, v in zip(bars, vals):
        if v == v: ax.text(b.get_x()+b.get_width()/2, v*1.03, f'{v:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(OPS, fontweight='bold')
ax.axhline(1, color='k', ls='--', alpha=0.5); ax.set_ylabel('speedup  (PyTorch ms / TinyMLP ms)')
ax.set_title(f'TinyMLP speedup over PyTorch  (hidden={H}, layers={L}, batch={B:,})', fontweight='bold')
ax.legend(); ax.grid(True, axis='y', which='both', alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_speedup.png'), dpi=110); plt.close(fig)

# ---------- Chart 4: speedup vs hidden dim (geomean over layers, train op) ----------
fig, ax = plt.subplots(figsize=(9, 5))
for model in MODELS:
    ys = [geomean([sp((model, 'train', H, L, B)) for L in LAYERS for B in BATCHES]) for H in HIDDENS]
    ax.plot(HIDDENS, ys, 'o-', lw=2.4, ms=9, label=f'{model.upper()} (train)')
ax.set_xscale('log', base=2); ax.set_xticks(HIDDENS); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.axhline(1, color='k', ls='--', alpha=0.5); ax.set_xlabel('hidden dim'); ax.set_ylabel('speedup (geomean over layers, batch)')
ax.set_title('End-to-end training speedup vs hidden dim', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_speedup_vs_dim.png'), dpi=110); plt.close(fig)

# ---------- report.md ----------
def md_table(rows, hdr):
    s = '| ' + ' | '.join(hdr) + ' |\n| ' + ' | '.join(['---']*len(hdr)) + ' |\n'
    for r in rows: s += '| ' + ' | '.join(str(c) for c in r) + ' |\n'
    return s

L = ['# TinyMLP vs PyTorch — Benchmark Report\n',
     '**GPU:** NVIDIA RTX 4060 Laptop (sm_89) · **PyTorch:** 2.9.0+cu126 (pure fp16) · **TinyMLP:** fused fp16 WMMA\n',
     'Latency in ms (lower is better); speedup = PyTorch / TinyMLP (>1 = TinyMLP faster). '
     'PyTorch hash grid = naive Instant-NGP-style encoding (the realistic eager baseline).\n']

# headline geomean speedups
L.append('## Headline — geometric-mean speedup across all configs\n')
rows = []
for model in MODELS:
    rows.append([model] + [f"{geomean([sp((model, op, H, Ly, B)) for H in HIDDENS for Ly in LAYERS for B in BATCHES]):.1f}x" for op in OPS])
L.append(md_table(rows, ['model'] + OPS) + '\n')

L.append('## Speedup at a representative config (hidden=64, layers=4, batch=1,048,576)\n')
rows = []
for model in MODELS:
    for op in OPS:
        k = (model, op, 64, 4, 1048576)
        rows.append([model, op, f"{tm.get(k, float('nan')):.3f}", f"{pt.get(k, float('nan')):.2f}", f"**{sp(k):.1f}x**"])
L.append(md_table(rows, ['model', 'op', 'TinyMLP ms', 'PyTorch ms', 'speedup']) + '\n')

L.append('## Charts\n')
for img, cap in [('fig_speedup.png', 'Speedup by op/model (batch 1M)'),
                 ('fig_mlp_latency.png', 'Plain MLP latency vs batch'),
                 ('fig_hashgrid_latency.png', 'Hash-grid MLP latency vs batch'),
                 ('fig_speedup_vs_dim.png', 'Training speedup vs hidden dim')]:
    L.append(f'**{cap}**\n\n![{cap}]({img})\n')

# full detail tables
L.append('## Full results\n')
for model in MODELS:
    L.append(f'### {model.upper()}\n')
    for op in OPS:
        L.append(f'**{op}** — ms (TinyMLP / PyTorch / speedup)\n')
        hdr = ['H/L \\ batch'] + [f'{B:,}' for B in BATCHES]
        rows = []
        for H in HIDDENS:
            for Ly in LAYERS:
                cells = []
                for B in BATCHES:
                    k = (model, op, H, Ly, B)
                    cells.append(f"{tm.get(k, float('nan')):.3f} / {pt.get(k, float('nan')):.2f} / {sp(k):.1f}x")
                rows.append([f'{H}/{Ly}'] + cells)
        L.append(md_table(rows, hdr) + '\n')

open(os.path.join(RES, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L))
print('Wrote charts + report.md to', RES)
print('\nHeadline geomean speedup (TinyMLP faster by):')
for model in MODELS:
    for op in OPS:
        print(f'  {model:9s} {op:10s} {geomean([sp((model, op, H, Ly, B)) for H in HIDDENS for Ly in LAYERS for B in BATCHES]):6.1f}x')
