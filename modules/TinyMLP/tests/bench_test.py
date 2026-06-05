import torch
import time

def calculate_metrics(B, H, L, ms):
    # Math Ops calculation (2 * M * N * K per layer)
    flops = 2.0 * L * B * H * H
    tflops = (flops / 1e12) / (ms / 1000.0)

    # Memory calculation matching your C++ exactly:
    # (2 * B * H * sizeof(half)) + (H * H * sizeof(half)) per layer
    bytes_moved = L * (2.0 * B * H * 2 + H * H * 2)
    gbs = (bytes_moved / 1e9) / (ms / 1000.0)
    
    return tflops, gbs

def run_standard_sweep():
    hr = "+---------+---------+--------+------------+----------+----------+"
    print(f"\n[Benchmark] warmup=10, iters=200")
    print(hr)
    print(f"| {'Batch':<7} | {'Hidden':<7} | {'Layers':<6} | {'ms / fwd':<10} | {'TFLOPS':<8} | {'GB/s':<8} |")
    print(hr)

    batches = [65536, 65536*2, 655368*4]
    hiddens = [32, 64, 128]
    layers_list = [1, 2, 3]

    warmup = 10
    iters = 50

    for L in layers_list:
        for H in hiddens:
            for B in batches:
                # 1. Setup Data (Everything in FP16 to satisfy PyTorch's strict eager mode)
                x = torch.randn(B, H, dtype=torch.float16, device='cuda')
                weights = [torch.randn(H, H, dtype=torch.float16, device='cuda') for _ in range(L)]
                biases = [torch.randn(H, dtype=torch.float16, device='cuda') for _ in range(L)]

                # 2. Forward Pass Definition
                def forward():
                    out = x
                    for i in range(L):
                        out = torch.nn.functional.linear(out, weights[i], biases[i])
                        if i < L - 1:
                            out = torch.nn.functional.relu(out)
                    return out

                # 3. Warmup
                for _ in range(warmup):
                    forward()
                torch.cuda.synchronize()

                # 4. Timed Loop
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(iters):
                    forward()
                end_event.record()
                torch.cuda.synchronize()

                ms_total = start_event.elapsed_time(end_event)
                ms = ms_total / iters
                
                tflops, gbs = calculate_metrics(B, H, L, ms)

                print(f"| {B:<7} | {H:<7} | {L:<6} | {ms:<10.3f} | {tflops:<8.4f} | {gbs:<8.2f} |")
            print(hr)

def run_massive_scale(B=1048576, H=256, L=16, warmup=5, iters=10):
    print("\n[Benchmark] Massive Scale Hardware Test")
    print(f"Config : B={B}  H={H}  L={L}  warmup={warmup}  iters={iters}")
    
    x = torch.randn(B, H, dtype=torch.float16, device='cuda')
    weights = [torch.randn(H, H, dtype=torch.float16, device='cuda') for _ in range(L)]
    biases = [torch.randn(H, dtype=torch.float16, device='cuda') for _ in range(L)]

    vram_mb = (B * H * 2 + B * H * 4) / (1024.0 * 1024.0)
    print(f"VRAM   : {vram_mb:.2f} MB allocated for I/O buffers")

    def forward():
        out = x
        for i in range(L):
            out = torch.nn.functional.linear(out, weights[i], biases[i])
            if i < L - 1:
                out = torch.nn.functional.relu(out)
        return out

    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()
    print("Warmup done.")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        forward()
    end_event.record()
    torch.cuda.synchronize()

    ms_total = start_event.elapsed_time(end_event)
    ms = ms_total / iters
    
    tflops, gbs = calculate_metrics(B, H, L, ms)
    
    print(f"Time   : {ms:.3f} ms / forward")
    print(f"Perf   : {tflops:.2f} TFLOPS  |  {gbs:.2f} GB/s")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False 
    
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    
    run_standard_sweep()
    run_massive_scale()