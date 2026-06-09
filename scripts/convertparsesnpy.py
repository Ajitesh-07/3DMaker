import os
import json
import numpy as np

def convert_llff_to_ngp(dataset_path, output_json_name="transforms_train.json"):
    npy_path = os.path.join(dataset_path, "poses_bounds.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Could not find {npy_path}")

    raw_data = np.load(npy_path)
    num_images = raw_data.shape[0]
    
    poses = raw_data[:, :15].reshape(-1, 3, 5)
    bounds = raw_data[:, 15:]
    
    H, W, focal = poses[0, 0, 4], poses[0, 1, 4], poses[0, 2, 4]
    
    images_dir = os.path.join(dataset_path, "images")
    folder_name = "images" 
    
    if not os.path.exists(images_dir):
        for factor in [4, 8]:
            alt_dir = os.path.join(dataset_path, f"images_{factor}")
            if os.path.exists(alt_dir):
                images_dir = alt_dir
                folder_name = f"images_{factor}"
                scale = factor
                H, W, focal = H / scale, W / scale, focal / scale
                break

    for f in os.listdir(images_dir):
        if f.endswith(('.JPG', '.JPEG', '.PNG')):
            old_path = os.path.join(images_dir, f)
            name, ext = os.path.splitext(f)
            new_path = os.path.join(images_dir, name + ext.lower())
            os.rename(old_path, new_path)

    img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(img_files) != num_images:
        print(f"Warning: Found {len(img_files)} images but {num_images} poses. Array mismatch might occur.")

    camera_angle_x = 2.0 * np.arctan(W / (2.0 * focal))

    out_json = {
        "camera_angle_x": float(camera_angle_x),
        "fl_x": float(focal),
        "fl_y": float(focal),
        "cx": float(W / 2.0),
        "cy": float(H / 2.0),
        "w": int(W),
        "h": int(H),
        "aabb_scale": 4, 
        "frames": []
    }

    for i in range(min(num_images, len(img_files))):
        c2w_llff = poses[i, :, :4] 
        
        c_down = c2w_llff[:, 0]
        c_right = c2w_llff[:, 1]
        c_back = c2w_llff[:, 2]
        t = c2w_llff[:, 3]
        
        c_up = -c_down
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, 0] = c_right
        transform_matrix[:3, 1] = c_up
        transform_matrix[:3, 2] = c_back
        transform_matrix[:3, 3] = t
        
        frame_data = {
            "file_path": f"{folder_name}/{img_files[i]}",
            "transform_matrix": transform_matrix.tolist()
        }
        out_json["frames"].append(frame_data)

    output_filepath = os.path.join(dataset_path, output_json_name)
    with open(output_filepath, 'w') as f:
        json.dump(out_json, f, indent=2)
        
    print(f"Successfully generated {output_json_name} mapping to folder '{folder_name}' with {len(out_json['frames'])} standardized frames.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:

        print("Usage: python script.py <path_to_scene_dir>")
        sys.exit(1)
    
    target_scene_dir = sys.argv[1]
    
    try:
        convert_llff_to_ngp(target_scene_dir)
    except Exception as e:
        print(f"Error during execution: {e}")
