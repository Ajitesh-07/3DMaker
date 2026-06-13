import os
import sys
import subprocess
import json
import math
import shutil

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
    run_cmd([colmap_exe, "feature_extractor", 
             "--database_path", db_path, 
             "--image_path", images_path])

    print("--- 2. Matching Features ---")
    run_cmd([colmap_exe, "exhaustive_matcher", 
             "--database_path", db_path])

    print("--- 3. Reconstructing 3D Map (SfM) ---")
    run_cmd([colmap_exe, "mapper", 
             "--database_path", db_path, 
             "--image_path", images_path, 
             "--output_path", sparse_path])

    # COLMAP mapper creates subfolders like "0", "1" for reconstructed models. We use "0".
    model_0_path = os.path.join(sparse_path, "0")
    if not os.path.exists(model_0_path):
        print("Error: COLMAP failed to reconstruct any models.")
        sys.exit(1)

    print("--- 4. Converting Binary Model to TXT ---")
    run_cmd([colmap_exe, "model_converter", 
             "--input_path", model_0_path, 
             "--output_path", sparse_txt_path, 
             "--output_type", "TXT"])

    print("--- 5. Parsing TXT to transforms.json ---")
    # Read cameras.txt
    camera_angle_x = 0.69 # Default fallback
    with open(os.path.join(sparse_txt_path, "cameras.txt"), "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 5: # ID MODEL WIDTH HEIGHT PARAMS...
                width = float(parts[2])
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

    # Align EVERY point first — we need the full cloud to estimate the background extent
    # (and therefore how many cascades the engine needs), not just the subject.
    all_aligned = [align_pt(p["pos"]) for p in points]

    # Subject = top 20% most-viewed points, which isolates the main object from background.
    subject_points = points
    if len(points) > 100:
        subject_points = sorted(points, key=lambda p: p["views"], reverse=True)[:int(len(points) * 0.20)]
    pts_aligned = [align_pt(p["pos"]) for p in subject_points]

    # Default: single cascade (object-centric / bounded scene). Overridden below if the
    # background reaches well beyond the subject.
    num_cascades = 1

    if len(pts_aligned) > 100:
        # Use median for robust center
        center = [
            sorted([p[0] for p in pts_aligned])[len(pts_aligned)//2],
            sorted([p[1] for p in pts_aligned])[len(pts_aligned)//2],
            sorted([p[2] for p in pts_aligned])[len(pts_aligned)//2]
        ]
        # Use 90th percentile distance for robust scaling (subject fills ~unit sphere -> cascade 0)
        dists = sorted([math.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2 + (p[2]-center[2])**2) for p in pts_aligned])
        obj_radius = dists[int(len(dists) * 0.90)]
        scale = 1.0 / (obj_radius + 1e-5)
        print(f"Scene bounds computed from Point Cloud. Object radius: {obj_radius:.2f}, Scale: {scale:.2f}")

        # Estimate how far the background extends relative to the subject. Cascade c covers a
        # power-of-2 box (~1.5 * 2^c in normalized units), so pick enough cascades to enclose
        # the 95th-percentile scene radius. ratio ~1 (turntable) -> 1 cascade; bigger room ->
        # more. Clamped to a sane maximum.
        if len(all_aligned) > 100:
            scene_dists = sorted(
                math.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2 + (p[2]-center[2])**2)
                for p in all_aligned
            )
            scene_radius = scene_dists[int(len(scene_dists) * 0.95)]
            ratio = scene_radius / (obj_radius + 1e-5)
            # Cascade 0 is the engine's base AABB, which already spans ~1.5x the unit subject
            # radius. Only background BEYOND that needs extra (power-of-2) cascades, so divide
            # out the base-box headroom before taking log2. ratio <= 1.5 -> everything fits in
            # cascade 0 -> num_cascades = 1.
            BASE_HALF_EXTENT = 1.5  # must match NerfOptions aabb half-width in Pipeline.cu
            needed = max(ratio / BASE_HALF_EXTENT, 1e-6)
            num_cascades = max(1, min(6, int(math.ceil(math.log2(needed))) + 1))
            print(f"Scene radius: {scene_radius:.2f} (ratio {ratio:.2f}) -> num_cascades = {num_cascades}")
    else:
        # Fallback to camera center
        cam_centers = [[frame["transform_matrix"][0][3], frame["transform_matrix"][1][3], frame["transform_matrix"][2][3]] for frame in frames]
        center = [sum(c[i] for c in cam_centers)/len(cam_centers) for i in range(3)]
        max_dist = max(math.sqrt((c[0]-center[0])**2 + (c[1]-center[1])**2 + (c[2]-center[2])**2) for c in cam_centers)
        scale = 1.0 / (max_dist + 1e-5)
        print("Warning: Point cloud sparse. Falling back to camera bounds.")

    for frame in frames:
        c2w = frame["transform_matrix"]
        # Shift translation
        c2w[0][3] = (c2w[0][3] - center[0]) * scale
        c2w[1][3] = (c2w[1][3] - center[1]) * scale
        c2w[2][3] = (c2w[2][3] - center[2]) * scale

    transforms = {
        "camera_angle_x": camera_angle_x,
        "num_cascades": num_cascades,
        "frames": frames
    }

    out_file = os.path.join(dataset_path, "transforms_train.json")
    with open(out_file, "w") as f:
        json.dump(transforms, f, indent=4)
        
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
