"""Plan a smooth fly-through camera path from a dataset's training cameras.

Follows the capture trajectory (frames in file-name order), because views far from any training
camera are where a NeRF looks worst. Real captures are jittery (Deep Blending drjohnson turns a
median 36 deg between consecutive photos), so positions are Gaussian-smoothed along the capture
order and resampled at constant speed, the camera looks along the direction of travel (yaw
smoothed over ~1 s), and it stays level (no roll, the capture's median pitch). Writes a transforms-style JSON for `3DMaker.exe render`:

    python scripts/flythrough_path.py data/drjohnson_fit            # -> data/drjohnson_fit/flythrough.json
    3DMaker.exe render --model data/drjohnson_fit/model.inerf --cameras data/drjohnson_fit/flythrough.json \
                       --out data/drjohnson_fit/flythrough           # -> flythrough/*.png + flythrough.mp4
"""
import argparse
import json
import os

import numpy as np


def gauss_smooth(x, sigma):
    """Gaussian-smooth rows of x along axis 0 (edge-padded so the ends don't shrink toward 0)."""
    if sigma <= 0:
        return x.copy()
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(xp[:, c], k, mode="valid") for c in range(x.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset folder with transforms_train.json")
    ap.add_argument("--out", default=None, help="output JSON (default <dataset>/flythrough.json)")
    ap.add_argument("--seconds", type=float, default=20.0, help="video length")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pos-sigma", type=float, default=6.0, help="position smoothing, in capture frames")
    ap.add_argument("--look-sigma", type=float, default=1.0, help="heading smoothing, in seconds")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()

    tj = json.load(open(os.path.join(a.dataset, "transforms_train.json")))
    frames = sorted(tj["frames"], key=lambda f: f["file_path"])
    M = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    C, U, F = M[:, :3, 3], M[:, :3, 1], -M[:, :3, 2]

    up = U.mean(0)
    up /= np.linalg.norm(up)
    pitch = np.median(np.arcsin(np.clip(F @ up, -1, 1)))        # keep the typical capture tilt

    Cs = gauss_smooth(C, a.pos_sigma)

    # Resample at constant speed along the smoothed path (arc length).
    seg = np.linalg.norm(np.diff(Cs, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(round(a.seconds * a.fps))
    t = np.linspace(0, s[-1], n)
    P = np.stack([np.interp(t, s, Cs[:, c]) for c in range(3)], 1)

    # Heading = direction of travel on the floor plane. Averaging the capture headings does not work:
    # photographers point every which way at one spot, so the average flips. Yaw is unwrapped and
    # smoothed over --look-sigma seconds so reversals become slow pans instead of cuts.
    e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    T = np.gradient(P, axis=0)
    yaw = np.unwrap(np.arctan2(T @ e2, T @ e1))
    yaw = gauss_smooth(yaw[:, None], a.look_sigma * a.fps)[:, 0]
    Hh = np.outer(np.cos(yaw), e1) + np.outer(np.sin(yaw), e2)

    out_frames = []
    for p, h in zip(P, Hh):
        f = np.cos(pitch) * h + np.sin(pitch) * up
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, u, -f, p
        out_frames.append({"transform_matrix": c2w.tolist()})

    out = a.out or os.path.join(a.dataset, "flythrough.json")
    # Same horizontal FOV as the capture camera; the vertical FOV follows from --width/--height.
    json.dump({"camera_angle_x": tj["camera_angle_x"], "w": a.width, "h": a.height, "frames": out_frames},
              open(out, "w"), indent=1)
    turn = np.degrees(np.arccos(np.clip((Hh[1:] * Hh[:-1]).sum(1), -1, 1)))
    print(f"{len(frames)} capture cameras -> {n} frames ({a.seconds:g} s at {a.fps} fps), path length {s[-1]:.2f}")
    print(f"pitch {np.degrees(pitch):.1f} deg, turn max {turn.max():.2f} / p95 {np.percentile(turn, 95):.2f} deg/frame")
    print("wrote", out)


if __name__ == "__main__":
    main()
