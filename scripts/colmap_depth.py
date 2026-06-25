"""
Sparse depth-prior extraction from a COLMAP TXT reconstruction.

For DS-NeRF-style geometry sharpening: for every COLMAP keypoint that triangulated
to a 3D point, emit (u, v, D, sigma) where

    D     = scale * ||X_cam||                 along-ray depth, renderer-normalized frame
    sigma = sigma_base * (1 + err / median_err)   relative-error widened Gaussian width

The math/units behind these are in:
  - docs/depth_supervision.md        (the depth loss this feeds)
  - docs/depth_uncertainty_sigma.md  (why sigma is set this way, not from raw pixels)

Two facts make D simple:
  * The renderer marches t as EUCLIDEAN distance along NORMALIZED ray directions
    (RenderKernels.cu:33), so the correct depth is ||X_cam|| (= X_cam.z / cos(theta),
    but needs no intrinsics). NOT the raw COLMAP z-component.
  * Only `scale` changes a distance. The OpenCV->OpenGL flip, the PCA alignment, and
    the recenter applied in colmap2nerf.py are all isometries, so they leave
    ||X_cam|| untouched -- we can work entirely in COLMAP's original frame and just
    multiply by `scale` at the end.
"""

import os
import struct
import numpy as np


def qvec2rotmat(qw, qx, qy, qz):
    """COLMAP world-to-camera quaternion -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz,     2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,     2*qy*qz + 2*qw*qx,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def _read_points3d(points3d_path):
    """Parse points3D.txt -> ({point3d_id: (xyz, error)}, median_error).

    Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[...]
    """
    pts = {}
    errors = []
    with open(points3d_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            pid = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            err = float(parts[7])
            pts[pid] = (xyz, err)
            if err > 0:
                errors.append(err)
    median_err = float(np.median(errors)) if errors else 1.0
    return pts, max(median_err, 1e-6)


def _read_images(images_path):
    """Yield (image_name, R_w2c, t_w2c, points2d) per registered image.

    images.txt stores TWO lines per image:
      line A: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      line B: X1 Y1 POINT3D_ID1  X2 Y2 POINT3D_ID2  ...   (may be empty)
    We strip only the comment header (NOT blank lines, since an image with no
    observations has an empty line B and dropping it would desync the pairing).
    """
    with open(images_path, "r") as f:
        content = [ln for ln in f.readlines() if not ln.startswith("#")]
    # A stray trailing newline at EOF would break the strict 2-line pairing.
    while content and content[-1].strip() == "":
        content.pop()

    for i in range(0, len(content) - 1, 2):
        pose = content[i].split()
        if len(pose) < 10:
            continue
        qw, qx, qy, qz = map(float, pose[1:5])
        tx, ty, tz = map(float, pose[5:8])
        name = pose[9]

        R = qvec2rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)

        toks = content[i + 1].split()
        points2d = [(float(toks[j]), float(toks[j + 1]), int(toks[j + 2]))
                    for j in range(0, len(toks) - 2, 3)]
        yield name, R, t, points2d


def extract_sparse_depth(sparse_txt_path, scale, sigma_base=0.02,
                         align_pt=None, center=None, max_radius=None):
    """Build per-image sparse depth priors from a COLMAP TXT model.

    Args:
      sparse_txt_path: dir holding images.txt + points3D.txt (COLMAP TXT export).
      scale:           the same recenter/scale factor baked into the poses by
                       colmap2nerf.py (puts D in the renderer's normalized frame).
      sigma_base:      base Gaussian width in normalized units (~0.01-0.03).
      align_pt:        optional fn mapping a raw COLMAP xyz -> the PCA-aligned frame
                       `center` lives in (colmap2nerf's align_pt closure). Required
                       for the out-of-grid cull below.
      center:          scene center in the aligned frame (colmap2nerf's `center`).
      max_radius:      cull radius in NORMALIZED units. A point whose normalized
                       distance from `center` exceeds this is dropped -- it falls
                       outside the renderer's outermost occupancy cascade, so no
                       rendering weights ever land at its depth and supervising it
                       would only drag mass toward the grid boundary. These far
                       points are also the low-parallax / least-reliable depths.
                       None / <=0 disables the cull (needs align_pt + center too).

    Returns:
      {image_name: [[u, v, D, sigma], ...]}   (u, v are COLMAP image-resolution px).
    """
    pts, median_err = _read_points3d(os.path.join(sparse_txt_path, "points3D.txt"))

    cull = (align_pt is not None and center is not None
            and max_radius is not None and max_radius > 0.0)
    if cull:
        center = np.asarray(center, dtype=np.float64)

    depth_map = {}
    total = 0
    n_far = 0
    for name, R, t, points2d in _read_images(os.path.join(sparse_txt_path, "images.txt")):
        samples = []
        for (u, v, pid) in points2d:
            if pid < 0 or pid not in pts:
                continue                          # unmatched feature -> no depth prior
            xyz, err = pts[pid]
            if cull:
                p_norm = scale * (np.asarray(align_pt(xyz), dtype=np.float64) - center)
                if float(np.linalg.norm(p_norm)) > max_radius:
                    n_far += 1
                    continue                       # outside the marchable grid -> useless prior
            x_cam = R @ xyz + t                    # world -> camera (COLMAP, +z forward)
            if x_cam[2] <= 0.0:
                continue                           # point is behind this camera
            D = scale * float(np.linalg.norm(x_cam))
            sigma = sigma_base * (1.0 + err / median_err)
            # Full precision -- the binary sidecar stores exact float32 (no rounding).
            samples.append([float(u), float(v), D, sigma])
        if samples:
            depth_map[name] = samples
            total += len(samples)

    culled = f", culled {n_far} beyond r={max_radius:.2f}" if cull else ""
    print(f"    Depth priors: {total} samples across {len(depth_map)} images "
          f"(median reproj err {median_err:.3f}px, sigma_base {sigma_base}{culled}).")
    return depth_map


# Binary record: u(uint16) v(uint16) D(float32) sigma(float32) = 12 bytes, little-endian.
DEPTH_REC_DTYPE = np.dtype([('u', '<u2'), ('v', '<u2'), ('D', '<f4'), ('s', '<f4')])
DEPTH_MAGIC = b'DPT1'
DEPTH_HEADER_FMT = '<4sIIII'                       # magic, version, num_frames, width, height
DEPTH_HEADER_SIZE = struct.calcsize(DEPTH_HEADER_FMT)  # 20 bytes


def write_depth_bin(out_path, depth_map, frame_names, width, height, version=1):
    """Pack per-frame sparse depth priors into a flat binary file with O(1) frame
    indexing and per-(u,v) byte-offset lookup.

    Layout (all little-endian; see DEPTH_* constants):

        header (20 B): magic 'DPT1' | version u32 | num_frames u32 | width u32 | height u32
        offsets:       (num_frames+1) u32 -- RECORD prefix-sum. Frame f owns records
                       [offsets[f], offsets[f+1]). Byte offset of record r is
                       DATA_START + r*12, with DATA_START = 20 + (num_frames+1)*4.
        records:       per frame, sorted ascending by key = v*width + u:
                       u u16 | v u16 | D f32 | sigma f32   (12 B each)

    To test a ray (frame f, u, v) for a prior: jump to frame f's record range via the
    offset table (O(1)), then binary-search that range by key -- pure byte-offset jumps,
    no scan. Pixel coords are floored to the integer pixel the renderer casts a ray
    through, and deduped to ONE record per (u,v), keeping the lowest-sigma (most
    reliable) hit. Frame blocks are emitted in `frame_names` order, which MUST match
    transforms.json frames[] so a frame index maps straight to its block.
    """
    if not (0 < width <= 65535 and 0 < height <= 65535):
        raise ValueError(f"image dims {width}x{height} out of uint16 range for (u,v)")

    per_frame = []
    total_in = 0
    for name in frame_names:
        best = {}                                   # (vi, ui) -> (D, sigma), min sigma wins
        for (u, v, D, sigma) in depth_map.get(name, []):
            total_in += 1
            ui = min(max(int(u), 0), width - 1)     # floor to the pixel index the ray-gen uses
            vi = min(max(int(v), 0), height - 1)
            prev = best.get((vi, ui))
            if prev is None or sigma < prev[1]:
                best[(vi, ui)] = (D, sigma)
        items = sorted(best.items(), key=lambda kv: kv[0][0] * width + kv[0][1])
        arr = np.empty(len(items), dtype=DEPTH_REC_DTYPE)
        for i, ((vi, ui), (D, sigma)) in enumerate(items):
            arr[i] = (ui, vi, D, sigma)
        per_frame.append(arr)

    counts = [len(a) for a in per_frame]
    offsets = np.zeros(len(counts) + 1, dtype='<u4')
    offsets[1:] = np.cumsum(counts, dtype=np.uint64)
    total = int(offsets[-1])

    header = struct.pack(DEPTH_HEADER_FMT, DEPTH_MAGIC, version, len(frame_names), width, height)
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(offsets.tobytes())
        for a in per_frame:
            f.write(a.tobytes())

    size_mb = (DEPTH_HEADER_SIZE + offsets.nbytes + total * DEPTH_REC_DTYPE.itemsize) / 1e6
    print(f"    Wrote {os.path.basename(out_path)}: {total} records / {len(frame_names)} frames "
          f"({size_mb:.2f} MB, deduped {total_in - total} dup pixels).")
    return total
