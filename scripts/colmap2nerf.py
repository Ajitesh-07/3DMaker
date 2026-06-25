import os
import sys
import subprocess
import json
import math
import shutil
import struct
import numpy as np

from colmap_depth import extract_sparse_depth, write_depth_bin

def qvec2rotmat(qw, qx, qy, qz):
    return [
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2]
    ]

def transpose(R):
    return [[R[j][i] for j in range(3)] for i in range(3)]

def mat_vec_mult(M, V):
    return [sum(M[i][j] * V[j] for j in range(3)) for i in range(3)]

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
        sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert COLMAP or Video to NeRF format")
    parser.add_argument("dataset_path", help="Path to images directory or a video file")
    parser.add_argument("--video_fps", type=int, default=8, help="FPS to extract from video")
    parser.add_argument("--depth_sigma", type=float, default=0.02,
                        help="Base Gaussian width (normalized units) for sparse depth priors; 0 disables")
    parser.add_argument("--depth_max_radius", type=float, default=0.0,
                        help="Cull depth priors farther than this (normalized units) from the scene "
                             "center; 0 = auto (outermost cascade box = 1.5 * 2^(num_cascades-1))")

    args = parser.parse_args()
    
    dataset_path = os.path.abspath(args.dataset_path)
    
    # Check if input is a video file
    if os.path.isfile(dataset_path):
        video_file = dataset_path
        dataset_path = os.path.splitext(video_file)[0] # e.g. "my_video"
        images_path = os.path.join(dataset_path, "images")
        os.makedirs(images_path, exist_ok=True)
        
        print(f"--- Extracting Video Frames to {images_path} ---")
        # Extract frames using ffmpeg
        run_cmd(["ffmpeg", "-i", video_file, "-qscale:v", "1", "-qmin", "1", "-vf", f"fps={args.video_fps}", os.path.join(images_path, "%04d.jpg")])
    else:
        images_path = os.path.join(dataset_path, "images")
        if not os.path.exists(images_path):
            print(f"Error: Expected 'images' folder inside {dataset_path}")
            sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    colmap_exe = os.path.join(project_root, "third_party", "colmap", "bin", "colmap.exe")

    if not os.path.exists(colmap_exe):
        print(f"Warning: Could not find COLMAP at {colmap_exe}. Trying global 'colmap' command.")
        colmap_exe = "colmap"

    colmap_temp = os.path.join(dataset_path, "colmap_temp")
    os.makedirs(colmap_temp, exist_ok=True)
    db_path = os.path.join(colmap_temp, "database.db")
    sparse_path = os.path.join(colmap_temp, "sparse")
    sparse_txt_path = os.path.join(colmap_temp, "sparse_txt")
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(sparse_txt_path, exist_ok=True)

    print("--- 1. Extracting Features ---")
    # single_camera=1: every frame is the SAME physical camera (handheld phone video), so COLMAP
    # should estimate ONE shared intrinsic set instead of per-image cameras. This is both more
    # accurate (all observations pooled) and makes our "read the first camera" parsing below valid.
    run_cmd([colmap_exe, "feature_extractor",
             "--database_path", db_path,
             "--image_path", images_path,
             "--ImageReader.single_camera", "1"])

    print("--- 2. Matching Features ---")
    run_cmd([colmap_exe, "exhaustive_matcher", 
             "--database_path", db_path])

    print("--- 3. Reconstructing 3D Map (SfM) ---")
    run_cmd([colmap_exe, "mapper", 
             "--database_path", db_path, 
             "--image_path", images_path, 
             "--output_path", sparse_path])

    # COLMAP mapper creates subfolders like "0", "1", ... one per reconstructed model. The
    # numbering is CREATION ORDER, not size — model "0" is frequently a tiny dead-end fragment
    # — so we select the model with the most registered images. A capture can fragment into
    # several disconnected models when tracking breaks (fast pans, blur, textureless stretches,
    # pure rotation with no parallax, unclosed loops); printing the breakdown makes that visible.
    def count_registered(model_dir):
        # COLMAP images.bin begins with a little-endian uint64 = number of registered images.
        try:
            with open(os.path.join(model_dir, "images.bin"), "rb") as f:
                return struct.unpack("<Q", f.read(8))[0]
        except Exception:
            return -1

    model_dirs = sorted(d for d in os.listdir(sparse_path)
                        if os.path.isdir(os.path.join(sparse_path, d)))
    if not model_dirs:
        print("Error: COLMAP failed to reconstruct any models.")
        sys.exit(1)

    num_extracted = len([f for f in os.listdir(images_path)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    model_counts = {d: count_registered(os.path.join(sparse_path, d)) for d in model_dirs}
    best_model = max(model_dirs, key=lambda d: model_counts[d])
    chosen_model_path = os.path.join(sparse_path, best_model)

    print(f"--- COLMAP produced {len(model_dirs)} model(s) from {num_extracted} extracted frames ---")
    total_registered = 0
    for d in model_dirs:
        n = model_counts[d]
        total_registered += max(n, 0)
        marker = "  <- USING THIS (largest)" if d == best_model else ""
        print(f"    model {d}: {n} registered images{marker}")
    best_n = model_counts[best_model]
    if len(model_dirs) > 1:
        print(f"    NOTE: capture fragmented into {len(model_dirs)} models; using the largest "
              f"(model {best_model}), discarding {total_registered - best_n} registered images "
              f"in the other model(s).")
    pct = 100.0 * best_n / max(num_extracted, 1)
    print(f"    Using model {best_model}: {best_n} / {num_extracted} frames ({pct:.0f}% of extracted).")

    print("--- 4. Converting Binary Model to TXT ---")
    run_cmd([colmap_exe, "model_converter",
             "--input_path", chosen_model_path,
             "--output_path", sparse_txt_path,
             "--output_type", "TXT"])

    print("--- 5. Parsing TXT to transforms.json ---")
    # Read cameras.txt
    camera_angle_x = 0.69 # Default fallback
    img_w, img_h = 0, 0   # native COLMAP image dims, for the depth sidecar's (u,v) grid
    with open(os.path.join(sparse_txt_path, "cameras.txt"), "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 5: # ID MODEL WIDTH HEIGHT PARAMS...
                width = float(parts[2])
                img_w = int(float(parts[2]))
                img_h = int(float(parts[3]))
                focal_length = float(parts[4])
                camera_angle_x = 2.0 * math.atan(width / (2.0 * focal_length))
                break

    # Read images.txt
    frames = []
    cam_centers = []
    
    with open(os.path.join(sparse_txt_path, "images.txt"), "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            if lines[i].startswith("#"): continue
            parts = lines[i].split()
            if len(parts) < 10: continue
            
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[9]

            # World to Camera
            R = qvec2rotmat(qw, qx, qy, qz)
            
            # Camera to World
            R_inv = transpose(R)
            T_inv = [-v for v in mat_vec_mult(R_inv, [tx, ty, tz])]
            cam_centers.append(T_inv)

            # Flip Y and Z axes (OpenCV -> OpenGL)
            c2w = [
                [R_inv[0][0], -R_inv[0][1], -R_inv[0][2], T_inv[0]],
                [R_inv[1][0], -R_inv[1][1], -R_inv[1][2], T_inv[1]],
                [R_inv[2][0], -R_inv[2][1], -R_inv[2][2], T_inv[2]],
                [0.0,         0.0,          0.0,          1.0]
            ]
            
            frames.append({
                "file_path": f"images/{image_name}",
                "transform_matrix": c2w
            })

    # Compute Average UP vector (Y-axis of cameras)
    up = [0.0, 0.0, 0.0]
    for frame in frames:
        up[0] += frame["transform_matrix"][0][1]
        up[1] += frame["transform_matrix"][1][1]
        up[2] += frame["transform_matrix"][2][1]
    
    up_norm = math.sqrt(up[0]**2 + up[1]**2 + up[2]**2)
    z_axis = [u / up_norm for u in up]
    
    if abs(z_axis[1]) > 0.99:
        x_axis = [1.0, 0.0, 0.0]
    else:
        x_axis = [-z_axis[2], 0.0, z_axis[0]]
        
    x_norm = math.sqrt(x_axis[0]**2 + x_axis[1]**2 + x_axis[2]**2)
    x_axis = [x / x_norm for x in x_axis]
    
    y_axis = [
        z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1],
        z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2],
        z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0]
    ]
    
    R_align = [x_axis, y_axis, z_axis]
    
    # Apply alignment to all frames
    for frame in frames:
        c2w = frame["transform_matrix"]
        new_c2w = [[0.0]*4 for _ in range(4)]
        new_c2w[3][3] = 1.0
        for r in range(3):
            for c in range(4):
                new_c2w[r][c] = R_align[r][0]*c2w[0][c] + R_align[r][1]*c2w[1][c] + R_align[r][2]*c2w[2][c]
        frame["transform_matrix"] = new_c2w
        
    # Parse points3D.txt for robust centering and scaling
    points = []
    points3d_path = os.path.join(sparse_txt_path, "points3D.txt")
    if os.path.exists(points3d_path):
        with open(points3d_path, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) >= 8:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    # parts[8:] are pairs of (IMAGE_ID, POINT2D_IDX)
                    num_views = (len(parts) - 8) // 2
                    points.append({"pos": [x, y, z], "views": num_views})
    
    # Align a single point to camera space (OpenCV -> OpenGL flip, then PCA alignment).
    def align_pt(pos):
        p_flip = [pos[0], -pos[1], -pos[2]]
        return [sum(R_align[r][c] * p_flip[c] for c in range(3)) for r in range(3)]

    # ---------- Robust scene normalization: center, scale, cascades ----------
    # Every point aligned (full cloud) — needed for the background-extent / cascade estimate.
    all_aligned = np.array([align_pt(p["pos"]) for p in points], dtype=np.float64) if points else np.zeros((0, 3))

    # Camera centers + forward (look) dirs in the ALIGNED frame (before the final shift below).
    cam_C = np.array([[fr["transform_matrix"][r][3] for r in range(3)] for fr in frames], dtype=np.float64)
    cam_F = np.array([[-fr["transform_matrix"][r][2] for r in range(3)] for fr in frames], dtype=np.float64)
    cam_F /= (np.linalg.norm(cam_F, axis=1, keepdims=True) + 1e-12)

    # Camera LOOK-AT CONVERGENCE: least-squares point closest to every camera view ray. For an
    # object orbit this IS the subject, and unlike the most-viewed-points median it is immune to
    # background point density (a busy floor/wall that is visible in most frames racks up view
    # counts and drags the median off the subject -> subject ends up jammed against the cube edge,
    # wasting grid resolution). Only meaningful when cameras look INWARD and the rays converge.
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, f in zip(cam_C, cam_F):
        P = np.eye(3) - np.outer(f, f)
        A += P; b += P @ c
    lookat, inward, cond = None, 0.0, float('inf')
    if len(cam_C) >= 3:
        try:
            lookat = np.linalg.solve(A, b)
            cond = float(np.linalg.cond(A))
            to_p = lookat - cam_C
            to_p /= (np.linalg.norm(to_p, axis=1, keepdims=True) + 1e-12)
            inward = float(np.mean(np.sum(cam_F * to_p, axis=1)))   # +1 inward (orbit), -1 outward (inside-out)
        except np.linalg.LinAlgError:
            lookat = None

    num_cascades = 1
    center = None
    obj_radius = None
    scale = None
    BASE_HALF_EXTENT = 1.5  # engine base AABB half-width (NerfOptions init in NerfTrainer.cu)

    # PREFERRED: object-orbit with a well-conditioned look-at -> center on the subject directly.
    if lookat is not None and inward > 0.3 and cond < 1e4 and len(all_aligned) > 0:
        center = lookat
        orbit_r = float(np.median(np.linalg.norm(cam_C - center, axis=1)))
        d_all = np.linalg.norm(all_aligned - center, axis=1)
        # Subject = points INSIDE the orbit (the object + its ground patch); this drops the far
        # floor/walls that wrecked the most-viewed median. Scale its 90th-pct radius -> ~unit.
        near = d_all[d_all < orbit_r]
        obj_radius = float(np.percentile(near, 90)) if near.size > 50 else 0.5 * orbit_r
        scale = 1.0 / (obj_radius + 1e-5)
        print(f"Centering: camera look-at convergence (object-orbit, inward={inward:.2f}, cond={cond:.1f}). "
              f"Object radius: {obj_radius:.2f}, Scale: {scale:.2f}")

    # FALLBACK: inside-out / non-convergent rays -> original most-viewed-points median.
    if center is None:
        subject_points = points
        if len(points) > 100:
            subject_points = sorted(points, key=lambda p: p["views"], reverse=True)[:int(len(points) * 0.20)]
        pts_aligned = np.array([align_pt(p["pos"]) for p in subject_points], dtype=np.float64) if subject_points else np.zeros((0, 3))
        if len(pts_aligned) > 100:
            center = np.median(pts_aligned, axis=0)
            obj_radius = float(np.percentile(np.linalg.norm(pts_aligned - center, axis=1), 90))
            scale = 1.0 / (obj_radius + 1e-5)
            print(f"Centering: most-viewed-points median (inside-out / non-convergent, inward={inward:.2f}). "
                  f"Object radius: {obj_radius:.2f}, Scale: {scale:.2f}")
        else:
            center = cam_C.mean(axis=0)
            obj_radius = float(np.max(np.linalg.norm(cam_C - center, axis=1)))
            scale = 1.0 / (obj_radius + 1e-5)
            print("Warning: Point cloud sparse. Falling back to camera bounds.")

    # How far the background reaches vs the subject -> how many power-of-2 cascades to enclose it.
    if len(all_aligned) > 100:
        scene_radius = float(np.percentile(np.linalg.norm(all_aligned - center, axis=1), 95))
        ratio = scene_radius / (obj_radius + 1e-5)
        # Cascade 0 is the engine's base AABB (~1.5x the unit subject radius); only background
        # BEYOND that needs extra cascades, so divide out the base-box headroom before log2.
        # ratio <= 1.5 -> everything fits in cascade 0 -> num_cascades = 1.
        needed = max(ratio / BASE_HALF_EXTENT, 1e-6)
        num_cascades = max(1, min(6, int(math.ceil(math.log2(needed))) + 1))
        print(f"Scene radius: {scene_radius:.2f} (ratio {ratio:.2f}) -> num_cascades = {num_cascades}")

    # Apply: shift translation by -center, then scale (rotation R_align already applied above).
    center = [float(center[0]), float(center[1]), float(center[2])]
    for frame in frames:
        c2w = frame["transform_matrix"]
        c2w[0][3] = (c2w[0][3] - center[0]) * scale
        c2w[1][3] = (c2w[1][3] - center[1]) * scale
        c2w[2][3] = (c2w[2][3] - center[2]) * scale

    # Sparse depth priors (DS-NeRF-style): per-keypoint [u, v, D, sigma] attached to
    # each frame. D uses the SAME `scale` as the poses, so it lands in the normalized
    # frame the renderer marches in. See scripts/colmap_depth.py + docs/depth_*.md.
    depth_file_name = None
    if args.depth_sigma > 0.0:
        try:
            if img_w <= 0 or img_h <= 0:
                raise ValueError("image width/height not found in cameras.txt")
            # Drop priors outside the renderer's outermost cascade box. Auto-bound =
            # base AABB half-extent * 2^(num_cascades-1) (the cube the engine marches);
            # --depth_max_radius overrides. align_pt + center put the cull in the same
            # normalized frame the poses were baked into.
            depth_max_radius = (args.depth_max_radius if args.depth_max_radius > 0.0
                                else BASE_HALF_EXTENT * (2 ** (num_cascades - 1)))
            depth_map = extract_sparse_depth(sparse_txt_path, scale, sigma_base=args.depth_sigma,
                                             align_pt=align_pt, center=center,
                                             max_radius=depth_max_radius)
            # Pack into a binary sidecar (colmap_depth.write_depth_bin): per-frame blocks
            # in frames[] order, records sorted by v*width+u for byte-offset (u,v) lookup.
            frame_names = [fr["file_path"].split("/", 1)[1] for fr in frames]
            write_depth_bin(os.path.join(dataset_path, "depth_train.bin"),
                            depth_map, frame_names, img_w, img_h)
            depth_file_name = "depth_train.bin"
        except Exception as e:
            print(f"Warning: sparse depth extraction failed ({e}); continuing without depth priors.")

    transforms = {
        "camera_angle_x": camera_angle_x,
        "num_cascades": num_cascades,
        "depth_sigma_base": args.depth_sigma,
        "frames": frames
    }

    # Train references the depth sidecar; test is held out (no supervision) so it omits it.
    train_transforms = dict(transforms)
    if depth_file_name:
        train_transforms["depth_file"] = depth_file_name

    out_file = os.path.join(dataset_path, "transforms_train.json")
    with open(out_file, "w") as f:
        json.dump(train_transforms, f, indent=4)

    # Also write a transforms_test.json for our test renderer
    out_file_test = os.path.join(dataset_path, "transforms_test.json")
    with open(out_file_test, "w") as f:
        json.dump(transforms, f, indent=4)

    print(f"--- 6. Cleanup & Success! ---")
    print(f"Saved {out_file}")
    print(f"Removing temporary files at {colmap_temp}...")
    shutil.rmtree(colmap_temp)
    print("Done! Dataset is ready for 3DMaker NeRF Training.")

if __name__ == "__main__":
    main()
