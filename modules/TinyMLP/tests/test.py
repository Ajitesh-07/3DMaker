import torch
import time

class StandardMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(torch.nn.Linear(in_d, hidden_dim, bias=True))
            
            # ReLU on all but the last layer
            if i < num_layers - 1:
                layers.append(torch.nn.ReLU())
                
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def profile_configuration(batch_size, hidden_dim, num_layers):
    device = torch.device('cuda')
    
    # Create model and move to FP16 on GPU
    model = StandardMLP(hidden_dim, hidden_dim, num_layers).half().to(device)
    
    # Fast initialization with 0s (Matches cudaMemset)
    x = torch.zeros((batch_size, hidden_dim), dtype=torch.float16, device=device)
    
    model.eval() # Disable any potential training overhead
    
    with torch.no_grad(): # Prevent PyTorch from allocating gradient memory
        # Warmup
        _ = model(x)
        torch.cuda.synchronize()
        
        # Dynamic run count matching C++
        num_runs = 50 if batch_size < 500000 else 10
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_runs):
            _ = model(x)
        end_event.record()
        torch.cuda.synchronize()
        
        total_ms = start_event.elapsed_time(end_event)
        avg_ms = total_ms / num_runs
        
    return avg_ms

def run_parameter_sweep():
    print("========================================================================")
    print("                PYTORCH BASELINE PROFILER                               ")
    print("========================================================================")
    print(f"{'Batch Size':<15} {'Hidden Dim':<15} {'Layers':<15} {'Time (ms)':<15} {'TFLOPS':<15}")
    print("-" * 72)

    dimensions = [32, 64, 128]

    for dim in dimensions:
        for layers in range(1, 11):
            batch = 1 << 16 # 2^16
            
            while batch <= (1 << 21): # up to 2^21
                try:
                    time_ms = profile_configuration(batch, dim, layers)
                    
                    # Same FLOP math as C++
                    total_flops = 2.0 * batch * dim * dim * layers
                    time_sec = time_ms / 1000.0
                    tflops = (total_flops / time_sec) / 1e12
                    
                    print(f"{batch:<15} {dim:<15} {layers:<15} {time_ms:<15.4f} {tflops:<15.2f}")
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"{batch:<15} {dim:<15} {layers:<15} {'OOM':<15} {'N/A':<15}")
                        torch.cuda.empty_cache()
                    else:
                        raise e
                
                batch <<= 1
            print("-" * 72)

if __name__ == "__main__":
    # Optimize PyTorch's cuBLAS backend selector
    torch.backends.cudnn.benchmark = True
    run_parameter_sweep()