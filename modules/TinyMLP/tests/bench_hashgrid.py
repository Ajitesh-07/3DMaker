import torch
import time

class PyTorchHashGrid(torch.nn.Module):
    def __init__(self, num_levels=16, features_per_level=2, log2_hashmap_size=19):
        super().__init__()
        self.num_levels = num_levels
        self.F = features_per_level
        self.T = 1 << log2_hashmap_size
        
        # Single embedding table for all levels (like Instant NGP)
        self.embeddings = torch.nn.Embedding(self.num_levels * self.T, self.F)
        
        # Base resolution and growth factor
        self.base_res = 16
        self.max_res = 2048
        self.b = (self.max_res / self.base_res) ** (1.0 / (self.num_levels - 1))
        
        # Primes for hashing
        self.primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long, device='cuda')
        
    def forward(self, x):
        # x: [Batch, 3] in [0, 1]
        B = x.shape[0]
        outputs = []
        
        for level in range(self.num_levels):
            res = int(self.base_res * (self.b ** level))
            
            # Scale coordinates
            x_scaled = x * res
            
            # 8 corners
            x_floor = torch.floor(x_scaled).long()
            x_frac = x_scaled - x_floor.float()
            
            level_offset = level * self.T
            
            # Trilinear interpolation weights
            c000 = (1 - x_frac[:, 0]) * (1 - x_frac[:, 1]) * (1 - x_frac[:, 2])
            c100 = x_frac[:, 0] * (1 - x_frac[:, 1]) * (1 - x_frac[:, 2])
            c010 = (1 - x_frac[:, 0]) * x_frac[:, 1] * (1 - x_frac[:, 2])
            c110 = x_frac[:, 0] * x_frac[:, 1] * (1 - x_frac[:, 2])
            c001 = (1 - x_frac[:, 0]) * (1 - x_frac[:, 1]) * x_frac[:, 2]
            c101 = x_frac[:, 0] * (1 - x_frac[:, 1]) * x_frac[:, 2]
            c011 = (1 - x_frac[:, 0]) * x_frac[:, 1] * x_frac[:, 2]
            c111 = x_frac[:, 0] * x_frac[:, 1] * x_frac[:, 2]
            
            weights = torch.stack([c000, c100, c010, c110, c001, c101, c011, c111], dim=1) # [B, 8]
            
            # Corner integer coordinates
            offsets = torch.tensor([[0,0,0], [1,0,0], [0,1,0], [1,1,0], 
                                    [0,0,1], [1,0,1], [0,1,1], [1,1,1]], device='cuda')
            
            corners = x_floor.unsqueeze(1) + offsets.unsqueeze(0) # [B, 8, 3]
            
            # Hash function: (x * p1 ^ y * p2 ^ z * p3) % T
            h = (corners[:, :, 0] * self.primes[0]) ^ \
                (corners[:, :, 1] * self.primes[1]) ^ \
                (corners[:, :, 2] * self.primes[2])
            
            # Fast modulo for power of 2
            indices = (h & (self.T - 1)) + level_offset # [B, 8]
            
            # Lookup
            vals = self.embeddings(indices) # [B, 8, F]
            
            # Interpolate
            interp = (vals * weights.unsqueeze(-1)).sum(dim=1) # [B, F]
            outputs.append(interp)
            
        return torch.cat(outputs, dim=-1) # [B, L*F]

def bench():
    print("Benchmarking Pure PyTorch Hash Grid...")
    device = 'cuda'
    model = PyTorchHashGrid().half().to(device)
    
    batch_sizes = [65536, 131072, 262144, 524288]
    
    for B in batch_sizes:
        x = torch.rand(B, 3, dtype=torch.float16, device=device)
        
        # Warmup
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()
        
        iters = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / iters
        items_per_sec = (B * 1000.0) / ms
        print(f"Batch: {B:<8} | Time: {ms:>8.2f} ms | Throughput: {items_per_sec:.2e} items/sec")

if __name__ == "__main__":
    bench()
