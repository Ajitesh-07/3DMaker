# 3DMaker

**3DMaker** is a zero-dependency, end-to-end Neural Radiance Field (NeRF) pipeline built entirely from scratch in C++ and CUDA. It is designed to take raw `.mp4` video files or image folders and automatically synthesize them into fully explorable, real-time 3D environments.

Inspired by architectures like Instant-NGP, 3DMaker uses a custom Multi-Resolution Hash Grid and a Hit-Centric Raymarcher to train models on consumer GPUs in a matter of seconds.

## ✨ Features

*   **Zero-Dependency Pipeline:** Simply point the engine at an `.mp4` video. The C++ orchestrator automatically invokes bundled FFmpeg and COLMAP binaries in the background to extract frames and compute 3D camera poses. No manual data preparation is required.
*   **Smart Point Cloud Centering:** Unlike standard NeRF scripts that center the world on the cameras (which breaks front-facing videos), 3DMaker parses the actual COLMAP `points3D.txt` point cloud, filters out background noise using view-track lengths, and perfectly centers the Neural Bounding Box directly on the physical object of interest. This guarantees maximum resolution and eliminates background "fog".
*   **Hit-Centric CUDA Engine:** Features a highly optimized, fully custom Digital Differential Analyzer (DDA) raymarcher. The engine uses an Occupancy Bitgrid to map empty space, allowing rays to instantly skip to the surface of the object.
*   **Real-Time 3D Viewer:** Includes a custom-built native OpenGL viewer (`3DViewer.exe`) with Zero-Copy CUDA Interop. The neural network renders pixels directly to the GPU display buffer (PBO) without ever copying data back to the CPU, allowing for high-FPS interactive orbiting and panning.

## 🏗️ Architecture & Modules

The codebase is split into modular components designed for high performance and separation of concerns:

### 1. `TinyMLP`
A high-performance, lightweight implementation of Multi-Layer Perceptrons and Multi-Resolution Hash Grids built entirely from the ground up in CUDA. This module serves as the core math engine, stripping away the bloat of standard deep learning frameworks like PyTorch or TensorFlow, and delivering **upwards of 100+ million multi-resolution evaluations per second**.
*   **Hardware Tensor Cores (FP16):** Utilizes custom `nvcuda::wmma` (Warp Matrix Multiply Accumulate) PTX instructions to perform fully fused matrix multiplication directly on the GPU's hardware Tensor Cores using native half-precision (FP16) mathematics for a 4x throughput boost.
*   **Asynchronous Memory Pipelines:** Uses Hopper/Ampere-grade `cp.async` (LDGSTS) instructions to bypass the L1 cache, prefetching weights from Global Memory directly into Shared Memory while the math units are actively calculating the previous matrix tile.
*   **Zero-Overhead Math:** Designed for absolute maximum throughput. Division and modulo operations across the entire network architecture are optimized using `uint32_t` indexing, forcing the `nvcc` compiler to synthesize single-instruction bitwise operations (`SHR` and `AND`), eliminating signed-integer division overhead.
*   **Dynamic Warp Folding:** The kernel dispatch logic intelligently folds the physical warp topology to perfectly match the neural network's hidden dimensions, ensuring 100% SM occupancy without thread divergence.
*   **Fused Adam Optimizer:** The weight update kernels inline the Adam optimization step directly during the backward pass backward, completely eliminating the need to store intermediate gradients in global memory and drastically reducing VRAM bandwidth.

### 2. `NeRFModule`
The mathematical heart of the project. Contains the CUDA kernels responsible for scene evaluation and orchestrating the rendering equations:
*   `InstantNerf.cu`: Manages the overall training loop, GPU memory allocation, and the Occupancy Grid. Uses an ultra-optimized **Hit-Centric Compaction Pipeline** that dynamically packs active rays in memory, slashing VRAM usage by over 90% compared to traditional worst-case ray allocators.
*   `DataLoader.cu`: Handles streaming massive datasets of rays from the CPU to the GPU using Asynchronous Pinned Memory, ensuring the GPU is never starved for data.
*   `processRays.cu`: The DDA raymarcher that physically computes where rays intersect the bounding box. It uses hierarchical bitfield traversal across multi-level occupancy grids to instantly skip empty space.

### 3. `3DMaker.exe` (The Orchestrator)
The command-line entry point for training. When executed, it:
1.  Ingests a video file.
2.  Runs `scripts/colmap2nerf.py` to orchestrate FFmpeg and COLMAP.
3.  Loads the resulting `transforms.json` into the `NeRFModule`.
4.  Trains the model.
5.  Automatically orbits a virtual camera around the finished scene to export a 360-degree high-quality `.mp4` showcase video.
6.  Saves the trained neural weights to `.inerf`.

### 4. `3DViewer.exe` (The Visualizer)
A standalone interactive viewer. It parses the original `transforms.json` to calculate the exact camera field-of-view (FOV) and reconstructs the starting angle of the very first frame. It features Blender-style camera controls (Middle Mouse pan, Left Click orbit).

## 🚀 How to Use

**1. Train a Video:**
Pass any video file to the training orchestrator. It will handle frame extraction, 3D reconstruction, and neural training automatically.
```powershell
.\build\Release\3DMaker.exe train --data data\my_video.mp4
```

**2. View the Result:**
Once training is complete, a `model.inerf` file is saved. You can explore it in real-time using the 3D Viewer:
```powershell
.\build\Release\3DViewer.exe data\my_video\transforms_train.json benchmarks\saved\model.inerf
```

*(Note: Ensure you are running from the `build` directory or your paths are correct relative to the executable).*

## 🐧 Linux Build Instructions

Building on Linux is fully supported using standard CMake. Ensure you have the **CUDA Toolkit (11.8 or higher)** and a modern C++ compiler (`g++-11` or higher) installed.

**1. Clone and Build the Main Project:**
```bash
git clone https://github.com/yourusername/3DMaker.git
cd 3DMaker
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**2. Compiling and Running TinyMLP Benchmarks:**
The high-performance core (`TinyMLP`) includes several standalone benchmarks to test hash-grid and matrix multiplication speeds. You can compile these by enabling the test flag:
```bash
cd 3DMaker/build
cmake -DCMAKE_BUILD_TYPE=Release -DTINYMLP_BUILD_TESTS=ON ..
make -j$(nproc)
```
Once compiled, you can run the benchmarks directly from the build directory. For example:
```bash
# Run the Multi-Resolution Hash Grid forward/backward pass benchmarks
./modules/TinyMLP/test_forward
./modules/TinyMLP/test_backward

# Run the raw Neural Network matrix multiplication benchmarks
./modules/TinyMLP/test_mlp
```
