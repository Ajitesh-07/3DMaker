import json, sys, math
import numpy as np

# Capture-geometry analyzer. The question that matters for free-viewpoint / mesh quality is
# "did the cameras go AROUND the subject" -- which is AZIMUTHAL ARC coverage, NOT view-direction
# concentration. View-dir concentration (R_fwd) conflates going-around with looking-down: a full
# orbit shot looking DOWN at the subject (drone, or circling a chair seat) has all view dirs on a
# tilted cone -> high R_fwd -> falsely reads "forward-facing". So azimuth is primary; R_fwd is a
# labelled diagnostic only.

def load(path):
    d = json.load(open(path))
    M = np.array([f["transform_matrix"] for f in d["frames"]], dtype=np.float64)
    C = M[:, :3, 3]
    fwd = -M[:, :3, 2]                       # OpenGL look dir = -z column
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    return d, C, fwd

def converge_point(C, fwd):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, f in zip(C, fwd):
        P = np.eye(3) - np.outer(f, f)
        A += P; b += P @ c
    try:
        return np.linalg.solve(A, b), float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return C.mean(0), float('inf')

def analyze(path):
    d, C, fwd = load(path)
    N = len(C)
    print(f"\n{'='*66}\n{path}\n{'='*66}")
    print(f"cameras: {N}   FOV: {math.degrees(d['camera_angle_x']):.1f} deg   num_cascades(json): {d.get('num_cascades')}")

    ctr = C.mean(0); Cc = C - ctr
    # orbit plane = the two largest-variance principal axes; normal = smallest
    w, V = np.linalg.eigh(Cc.T @ Cc)         # ascending eigenvalues
    n, e1, e2 = V[:, 0], V[:, 1], V[:, 2]

    # [1] AZIMUTHAL ARC around the subject (primary "did you go around")
    az = np.sort(np.degrees(np.arctan2(Cc @ e2, Cc @ e1)))
    gaps = np.diff(np.concatenate([az, [az[0] + 360]]))
    azimuth = 360 - gaps.max()
    print(f"\n[1] AZIMUTHAL ARC (primary): {azimuth:.0f} deg covered   (largest gap {gaps.max():.0f} deg)")
    print(f"    360 orbit ~ 330+,  partial arc ~ 150-300,  forward-facing < 150")

    # [2] LATITUDE / elevation coverage (pole coverage): viewing latitude spread about look-at point
    p, cond = converge_point(C, fwd)
    rel = C - p; dist = np.linalg.norm(rel, axis=1)
    lat = np.degrees(np.arcsin(np.clip((rel @ n) / (dist + 1e-12), -1, 1)))
    lat_range = lat.max() - lat.min()
    print(f"\n[2] LATITUDE SPREAD (pole coverage): {lat_range:.0f} deg "
          f"(min {lat.min():.0f}, max {lat.max():.0f})")
    print(f"    wide (>45) sees top+sides+under; narrow band misses top/underside")

    # [3] MODE: object-orbit (look inward) vs inside-out pano (look outward)
    to_p = p - C; to_p /= np.linalg.norm(to_p, axis=1, keepdims=True) + 1e-12
    inward = float(np.mean(np.sum(fwd * to_p, axis=1)))   # +1 inward, -1 outward
    mode = ("object-orbit (looking inward)" if inward > 0.3 else
            "inside-out pano (looking outward)" if inward < -0.3 else "mixed / flat")
    print(f"\n[3] MODE: {mode}   (inward score {inward:+.2f})")

    # [4] parallax + diagnostics
    axis_std = np.sqrt(np.maximum(np.sort(w)[::-1], 0) / N)
    Rf = np.linalg.norm(fwd.mean(0)); tilt = math.degrees(math.acos(min(1.0, Rf)))
    print(f"\n[4] diagnostics: parallax(baseline/depth)={axis_std[0]/(dist.mean()+1e-9):.2f}  "
          f"look-at cond={cond:.1f}")
    print(f"    R_fwd={Rf:.2f} (view-dirs ~{tilt:.0f} deg off cone axis) "
          f"-- HIGH R_fwd = steep/downward look, NOT necessarily forward-facing")

    # A full sphere needs the viewing latitude to cross the equator (subject seen from above AND
    # below its own middle), not merely a wide band on one side. A one-sided band = one pole / the
    # underside is never seen -> fine for free-viewpoint at that band, weak for a watertight mesh.
    one_sided = (lat.min() > 10) or (lat.max() < -10)

    # VERDICT (azimuth-primary)
    print(f"\n[VERDICT]")
    if azimuth >= 300:
        head = "FULL 360 azimuth" if inward > 0.3 else "FULL surround (inside-out pano)"
        if inward > 0.3 and (one_sided or lat_range < 30):
            side = "underside" if lat.max() < -10 else "top" if lat.min() > 10 else "top/underside"
            print(f"    {head}, but latitude band is ONE-SIDED ({lat.min():.0f}..{lat.max():.0f} deg) "
                  f"-> {side} never seen.")
            print(f"    Solid free-viewpoint at the captured band; for a watertight MESH also shoot from "
                  f"below/above (cross the equator).")
        else:
            print(f"    {head} + latitude crosses equator -> well-constrained for free-viewpoint and meshing.")
    elif azimuth >= 150:
        print(f"    PARTIAL arc (~{azimuth:.0f} deg) -- one side under-covered; off-arc views will degrade.")
    else:
        print(f"    NARROW / FORWARD-FACING (~{azimuth:.0f} deg) -- geometry under-constrained.")
        print(f"    High in-cone test PSNR but no free-viewpoint 3D / mesh. Orbit the subject.")

for p in sys.argv[1:]:
    try:
        analyze(p)
    except Exception as e:
        print(f"{p}: ERROR {e}")
