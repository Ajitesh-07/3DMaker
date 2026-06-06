import torch
import torch.nn as nn
import time

class MLP(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def run_performance_sweep():
    print("\n=========================================================================================")
    print("                             PYTORCH PERFORMANCE SWEEP")
    print("=========================================================================================")
    print("Dim\tLayers\tBatch Size\tTime/Step (ms)\tThroughput (GB/s)\tThroughput (items/sec)")
    print("-----------------------------------------------------------------------------------------")

    dims = [32]
    batch_sizes = [1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20]
    num_steps = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.cuda.amp.GradScaler()

    for dim in dims:
        for layers in range(2, 4):
            for batch_size in batch_sizes:
                model = MLP(dim, layers).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                criterion = nn.MSELoss()

                # Generate random data (FP16)
                inputs = torch.randn(batch_size, dim, dtype=torch.float16, device=device)
                targets = torch.randn(batch_size, dim, dtype=torch.float16, device=device)

                # Warmup
                for _ in range(5):
                    optimizer.zero_grad()
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                torch.cuda.synchronize()
                start_time = time.time()

                for _ in range(num_steps):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                torch.cuda.synchronize()
                end_time = time.time()

                total_ms = (end_time - start_time) * 1000.0
                ms_per_step = total_ms / num_steps
                
                throughput_items = (batch_size * 1000.0) / ms_per_step
                
                # Approximate GB/s calculation (same logic as C++ for apples-to-apples)
                bytes_per_step = batch_size * (dim * 2 + dim * 2 * 3) + batch_size * dim * 2 * layers * 2
                gbps = (bytes_per_step * 1000.0) / (ms_per_step * 1e9)

                print(f"{dim}\t{layers}\t{batch_size}\t\t{ms_per_step:g}\t\t{gbps:g}\t\t{throughput_items:g}")

                # Cleanup to avoid OOM
                del model
                del optimizer
                del inputs
                del targets
                del outputs
                del loss
                torch.cuda.empty_cache()

if __name__ == '__main__':
    run_performance_sweep()