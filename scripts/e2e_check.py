"""End-to-end smoke test: images/video -> COLMAP -> NeRF training -> renders, with stage timings
and a render-vs-photo comparison.

    python scripts/e2e_check.py <dataset_dir | video.mp4> [--steps N] [-- extra 3DMaker flags]

Runs the built 3DMaker binary exactly as a user would (with --validate), timestamps every stage it
prints, then compares frames/view_{0,1,2}.png against the matching input photos. Writes into the
dataset folder:
    e2e_log.txt        full 3DMaker output with elapsed-time prefixes
    e2e_summary.json   stage durations, registered images, held-out PSNR, per-view PSNR
    e2e_compare.png    photo | render | abs-error heatmap, one row per view
With --validate the engine holds out every 8th image, so view_0 is a held-out (unseen) camera
and views 1-2 are training cameras.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Visual Studio (multi-config) puts binaries in build/Release; Makefiles/Ninja on Linux in build/.
EXE = next((p for p in (os.path.join(ROOT, "build", "Release", "3DMaker.exe"),
                        os.path.join(ROOT, "build", "3DMaker")) if os.path.exists(p)),
           os.path.join(ROOT, "build", "Release", "3DMaker.exe"))

# (marker substring in 3DMaker / colmap2nerf output, stage that STARTS there)
STAGES = [
    ("Extracting Video Frames", "frame extraction"),
    ("--- 1. Extracting Features", "COLMAP features"),
    ("--- 2. Matching Features", "COLMAP matching"),
    ("--- 3. Reconstructing", "COLMAP mapping"),
    ("--- 4. Converting", "COLMAP export + depth priors"),
    ("Starting 3DMaker NeRF Training", "NeRF load + train"),
    ("HELD-OUT VALIDATION PSNR", "validation"),
    ("view_0.png", "render views + save"),   # not "Saved ": colmap2nerf also prints "Saved <json>"
]


def run(dataset, steps, extra):
    log_path = os.path.join(dataset if os.path.isdir(dataset) else os.path.splitext(dataset)[0], "e2e_log.txt")
    cmd = [EXE, "train", "--data", dataset, "--steps", str(steps), "--validate"] + extra
    print("Running:", " ".join(cmd), flush=True)
    t0 = time.time()
    marks, lines = [], []
    # PYTHONUNBUFFERED: colmap2nerf.py (launched by 3DMaker) would otherwise block-buffer its prints
    # into the pipe, so its stage markers would arrive late and skew the stage timings.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    buf = b""
    last_bar = 0.0
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch not in (b"\n", b"\r"):
            buf += ch
            continue
        line = buf.decode("utf-8", "replace").rstrip()
        buf = b""
        if not line:
            continue
        el = time.time() - t0
        lines.append(f"[{el:8.1f}s] {line}")
        for marker, stage in STAGES:
            if marker in line and not any(s == stage for s, _ in marks):
                marks.append((stage, el))
                print(f"[{el:7.1f}s] >>> {stage}", flush=True)
        is_bar = line.startswith("[#") or line.startswith("[ ")
        if is_bar:  # progress bar: print at most every 30 s
            if el - last_bar > 30:
                print(f"[{el:7.1f}s] {line.strip()}", flush=True)
                last_bar = el
        elif any(k in line for k in ("registered", "Using model", "PSNR", "Error", "error", "num_cascades",
                                     "Scene radius", "Centering", "Loaded", "Saved")):
            print(f"[{el:7.1f}s] {line.strip()}", flush=True)
    code = proc.wait()
    total = time.time() - t0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return code, total, marks, lines, log_path


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return -10.0 * np.log10(max(mse, 1e-10))


def compare(dataset_dir):
    tj = json.load(open(os.path.join(dataset_dir, "transforms_train.json")))
    rows, per_view = [], []
    for i in range(3):
        rp = os.path.join(dataset_dir, "frames", f"view_{i}.png")
        if not os.path.exists(rp) or i >= len(tj["frames"]):
            continue
        gt_path = os.path.join(dataset_dir, tj["frames"][i]["file_path"])
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        rd = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255.0
        if gt.shape != rd.shape:
            gt = np.asarray(Image.open(gt_path).convert("RGB").resize(rd.shape[1::-1]), dtype=np.float32) / 255.0
        err = np.abs(gt - rd).mean(axis=2)
        # "hot" heatmap: black = exact, red = 0.25 abs error, yellow = 0.5+
        heat = np.stack([np.clip(err * 4, 0, 1), np.clip(err * 4 - 1, 0, 1), np.zeros_like(err)], axis=2)
        rows.append(np.concatenate([gt, rd, heat], axis=1))
        per_view.append({"view": i, "photo": os.path.basename(gt_path),
                         "held_out": i % 8 == 0, "psnr_db": round(psnr(gt, rd), 2)})
    if rows:
        img = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(dataset_dir, "e2e_compare.png"))
    return per_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--steps", type=int, default=50000)
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]
    if not os.path.exists(EXE):
        sys.exit(f"Build first: {EXE} not found")

    code, total, marks, lines, log_path = run(args.dataset, args.steps, extra)
    ds = args.dataset if os.path.isdir(args.dataset) else os.path.splitext(args.dataset)[0]

    stages = {}
    for k, (name, t) in enumerate(marks):
        t_end = marks[k + 1][1] if k + 1 < len(marks) else total
        stages[name] = round(t_end - t, 1)
    text = "\n".join(lines)
    reg = re.search(r"Using model \S+: (\d+) / (\d+) frames", text)
    val = re.search(r"HELD-OUT VALIDATION PSNR: ([\d.]+)", text)
    ms = re.findall(r"([\d.]+) ms/step", text)
    summary = {
        "exit_code": code,
        "total_s": round(total, 1),
        "stages_s": stages,
        "registered_images": f"{reg.group(1)} / {reg.group(2)}" if reg else "n/a (poses existed)",
        "heldout_psnr_db": float(val.group(1)) if val else None,
        "final_ms_per_step": float(ms[-1]) if ms else None,
        "views": compare(ds) if code == 0 else [],
        "log": log_path,
    }
    with open(os.path.join(ds, "e2e_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    sys.exit(code)


if __name__ == "__main__":
    main()
