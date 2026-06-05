import torch
import time

def run_benchmark():
    print("========== Device Info ==========")
    print(f"Device                       : {torch.cuda.get_device_name()}")
    print("=================================\n")

    batch_sizes = [1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21]
    dims = [16, 32, 64, 128]
    layer_counts = [2, 4, 6, 8, 10]

    print("Starting PyTorch Profiling Sweep...\n")
    print(f"{'Batch Size':<12} {'Dim':<8} {'Layers':<10} {'Time (ms)':<15} {'TFLOPS':<15} Status")
    print("-" * 75)

    for batch_size in batch_sizes:
        for dim in dims:
            for num_layers in layer_counts:
                try:
                    # Allocate inputs
                    x = torch.randn(batch_size, dim, dtype=torch.float16, device='cuda', requires_grad=True)
                    
                    # Allocate layers (weights and biases)
                    layers = []
                    for _ in range(num_layers):
                        w = torch.randn(dim, dim, dtype=torch.float16, device='cuda', requires_grad=True)
                        b = torch.randn(dim, dtype=torch.float16, device='cuda', requires_grad=True)
                        layers.append((w, b))
                        
                    # Dummy gradient to jumpstart the backward pass (acting as d_loss_output)
                    grad_output = torch.randn(batch_size, dim, dtype=torch.float16, device='cuda')
                    
                    def forward_pass():
                        curr = x
                        for i, (w, b) in enumerate(layers):
                            # PyTorch linear is x @ w.T + b
                            curr = torch.matmul(curr, w.t()) + b
                            if i < num_layers - 1:
                                curr = torch.relu(curr)
                        return curr
                        
                    # 1. Warmup
                    out = forward_pass()
                    out.backward(gradient=grad_output)
                    torch.cuda.synchronize()
                    
                    # 2. Benchmark (Timing ONLY the backward pass)
                    runs = 10
                    times = []
                    
                    for _ in range(runs):
                        # Zero gradients
                        x.grad = None
                        for w, b in layers:
                            w.grad = None
                            b.grad = None
                            
                        # Rebuild the graph
                        out = forward_pass()
                        torch.cuda.synchronize()
                        
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        
                        start_event.record()
                        # Execute the backward pass
                        out.backward(gradient=grad_output)
                        end_event.record()
                        
                        torch.cuda.synchronize()
                        times.append(start_event.elapsed_time(end_event))
                        
                    avg_ms = sum(times) / runs
                    
                    # 3. Compute TFLOPS
                    # dW = 2 * batch * dim^2 (for all layers)
                    # dX = 2 * batch * dim^2 (for numLayers - 1 layers)
                    # Total FLOPS = (4 * numLayers - 2) * batch * dim^2
                    flops = (4 * num_layers - 2) * batch_size * (dim ** 2)
                    tflops = (flops / (avg_ms / 1000.0)) / 1e12
                    
                    print(f"{batch_size:<12} {dim:<8} {num_layers:<10} {avg_ms:<15.3f} {tflops:<15.2f} OK")
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"{batch_size:<12} {dim:<8} {num_layers:<10} {'N/A':<15} {'N/A':<15} OOM (Out of VRAM)")
                
                finally:
                    # Aggressive cleanup to prevent cascading OOMs
                    x = None
                    layers = None
                    grad_output = None
                    out = None
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_benchmark()