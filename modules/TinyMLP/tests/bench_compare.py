# PyTorch side of the TinyMLP-vs-PyTorch comparison.
# Emits the same CSV schema as bench_compare.cu:  framework,model,op,hidden,layers,batch,ms
# Pure fp16 throughout (tensor cores) = the fastest/most-charitable PyTorch baseline.
import torch, sys

torch.backends.cuda.matmul.allow_tf32 = False
DEV = 'cuda'
WARMUP, ITERS = 8, 40

# ---- pure-PyTorch Instant-NGP-style hash grid encoding (16 levels x 2 features -> 32) ----
class PyTorchHashGrid(torch.nn.Module):
    def __init__(self, num_levels=16, features_per_level=2, log2_hashmap_size=19):
        super().__init__()
        self.num_levels = num_levels; self.F = features_per_level; self.T = 1 << log2_hashmap_size
        self.embeddings = torch.nn.Embedding(self.num_levels * self.T, self.F)
        self.base_res = 16; self.b = (2048 / 16) ** (1.0 / (self.num_levels - 1))
        self.register_buffer('primes', torch.tensor([1, 2654435761, 805459861], dtype=torch.long))
        self.register_buffer('offsets', torch.tensor(
            [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=torch.long))
    def forward(self, x):
        outs = []
        for level in range(self.num_levels):
            res = int(self.base_res * (self.b ** level))
            xs = x * res
            xf = torch.floor(xs).long(); fr = xs - xf.float()
            c = [ (1-fr[:,0])*(1-fr[:,1])*(1-fr[:,2]), fr[:,0]*(1-fr[:,1])*(1-fr[:,2]),
                  (1-fr[:,0])*fr[:,1]*(1-fr[:,2]),      fr[:,0]*fr[:,1]*(1-fr[:,2]),
                  (1-fr[:,0])*(1-fr[:,1])*fr[:,2],      fr[:,0]*(1-fr[:,1])*fr[:,2],
                  (1-fr[:,0])*fr[:,1]*fr[:,2],          fr[:,0]*fr[:,1]*fr[:,2] ]
            w = torch.stack(c, dim=1)
            corners = xf.unsqueeze(1) + self.offsets.unsqueeze(0)
            h = (corners[:,:,0]*self.primes[0]) ^ (corners[:,:,1]*self.primes[1]) ^ (corners[:,:,2]*self.primes[2])
            idx = (h & (self.T - 1)) + level * self.T
            vals = self.embeddings(idx)
            outs.append((vals * w.unsqueeze(-1)).sum(dim=1))
        return torch.cat(outs, dim=-1)

def build_mlp(H, L):
    layers = []
    for i in range(L):
        layers.append(torch.nn.Linear(H, H))
        if i < L-1: layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers).half().to(DEV)

class HashNet(torch.nn.Module):
    def __init__(self, H, L, out=16):
        super().__init__()
        self.enc = PyTorchHashGrid()
        layers = [torch.nn.Linear(32, H), torch.nn.ReLU()]
        for _ in range(L-2): layers += [torch.nn.Linear(H, H), torch.nn.ReLU()]
        layers += [torch.nn.Linear(H if L>1 else 32, out)]
        self.head = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.head(self.enc(x).half())

def time_ms(fn, budget_ms=300.0, min_iters=3, max_iters=200):
    # adaptive: probe one iter, then run enough iters to fill ~budget_ms (bounds wall-clock
    # for slow ops like the naive PyTorch hashgrid while keeping fast ops well-sampled).
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    probe = max(s.elapsed_time(e), 1e-3)
    iters = int(min(max_iters, max(min_iters, budget_ms / probe)))
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def bench(model, op, H, L, B):
    if model == 'mlp':
        net = build_mlp(H, L); x = torch.randn(B, H, dtype=torch.float16, device=DEV)
        tgt = torch.randn(B, H, dtype=torch.float16, device=DEV)
    else:
        net = HashNet(H, L).half().to(DEV); x = torch.rand(B, 3, dtype=torch.float16, device=DEV)
        tgt = torch.randn(B, 16, dtype=torch.float16, device=DEV)

    if op == 'inference':
        net.eval()
        with torch.no_grad():
            ms = time_ms(lambda: net(x))
    elif op == 'backward':
        out = net(x); loss = ((out.float() - tgt.float())**2).mean()
        ms = time_ms(lambda: loss.backward(retain_graph=True))
    else:  # train
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        def step():
            opt.zero_grad(set_to_none=True)
            out = net(x); loss = ((out.float() - tgt.float())**2).mean()
            loss.backward(); opt.step()
        ms = time_ms(step)
    del net, x, tgt; torch.cuda.empty_cache()
    return ms

if __name__ == "__main__":
    # global warmup: absorb one-time CUDA/cuBLAS init + GPU clock ramp so config #1 isn't cold
    bench('mlp', 'train', 64, 2, 65536)
    bench('hashgrid', 'train', 64, 2, 65536)
    torch.cuda.synchronize()
    print("framework,model,op,hidden,layers,batch,ms")
    for H in [32, 64, 128]:
        for L in [2, 4]:
            for B in [65536, 262144, 1048576]:
                for model in ['mlp', 'hashgrid']:
                    for op in ['inference', 'backward', 'train']:
                        try:
                            ms = bench(model, op, H, L, B)
                            print(f"pytorch,{model},{op},{H},{L},{B},{ms:.5f}")
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"pytorch,{model},{op},{H},{L},{B},nan")
                        sys.stdout.flush()
