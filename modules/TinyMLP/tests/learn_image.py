import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

# Set random seed for reproducibility
torch.manual_seed(42)

class PyTorchTinyMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=8, num_layers=8):
        super().__init__()
        layers = []
        
        # Layer 0 (Input -> Hidden)
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Layers 1 to num_layers-2 (Hidden -> Hidden)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        # Layer num_layers-1 (Hidden -> Output)
        layers.append(nn.Linear(hidden_dim, output_dim))
        # No activation on final output based on the C++ script
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_dataset_from_image(filename, input_dim, device):
    img = Image.open(filename).convert('RGB')
    width, height = img.size
    batch_size = width * height
    
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = torch.tensor(img_data).view(-1, 3)
    
    # Pad targets to output_dim (8) to match the custom kernel layout
    targets = torch.zeros((batch_size, 8), dtype=torch.float32)
    targets[:, :3] = img_data
    
    # Replicate C++ Positional Encoding (Fourier Features) exactly
    num_freqs = (input_dim - 2) // 4
    
    x_coords = torch.arange(width)
    y_coords = torch.arange(height)
    gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    nx = (gx.flatten().float() / width) * 2.0 - 1.0
    ny = (gy.flatten().float() / height) * 2.0 - 1.0
    
    nx = nx.unsqueeze(1)
    ny = ny.unsqueeze(1)
    
    features = [nx, ny]
    for f in range(num_freqs):
        freq = (2.0 ** f) * math.pi
        features.append(torch.sin(nx * freq))
        features.append(torch.cos(nx * freq))
        features.append(torch.sin(ny * freq))
        features.append(torch.cos(ny * freq))
        
    inputs = torch.cat(features, dim=1)
    
    # Pad remaining input dimensions with zeros
    if inputs.shape[1] < input_dim:
        padding = torch.zeros((batch_size, input_dim - inputs.shape[1]))
        inputs = torch.cat([inputs, padding], dim=1)
        
    print(f"Loaded Image: {width}x{height} ({batch_size} pixels)")
    
    # Move to device and convert to half precision to mimic the C++ d_inputs layout
    return inputs.to(device).half(), targets.to(device).half(), width, height

def save_predicted_image(filename, outputs, width, height):
    # Clamp between 0 and 1, extract the RGB channels
    rgb = torch.clamp(outputs[:, :3], 0.0, 1.0).view(height, width, 3)
    rgb = (rgb.cpu().numpy() * 255.0).astype(np.uint8)
    
    img = Image.fromarray(rgb)
    img.save(filename, quality=100)

def main():
    print("--- End-to-End PyTorch Training Test ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Architecture params
    input_dim = 32
    hidden_dim = 32
    output_dim = 8
    num_layers = 4
    
    # Training Hyperparams
    num_steps = 1000
    lr = 3e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    loss_scale = 128.0
    
    os.makedirs("../images", exist_ok=True)
    
    inputs, targets, width, height = load_dataset_from_image("../images/image1.jpg", input_dim, device)
    batch_size = width * height
    
    model = PyTorchTinyMLP(input_dim, hidden_dim, output_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
    
    # Using reduction='sum' to manually divide by batchSize and match the C++ loss scaling
    criterion = nn.MSELoss(reduction='sum')
    
    # Modern PyTorch API for AMP scaling
    scaler = torch.amp.GradScaler('cuda', init_scale=loss_scale)
    
    print(f"Starting Training Loop... ({num_steps} steps)")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Quick warmup to compile graphs and prevent CUDA context initialization from skewing time metrics
    for _ in range(3):
        with torch.amp.autocast('cuda'):
            out = model(inputs)
            loss = criterion(out, targets) / output_dim
        scaler.scale(loss).backward()
        optimizer.zero_grad(set_to_none=True)
        
    torch.cuda.synchronize()
    start_event.record()
    
    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass in FP16
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            # C++ computes sum of squares across all 8 dims, divide by output_dim to match Mean over Features, Sum over Batch
            loss = criterion(outputs, targets) / output_dim 
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step == 1 or step % 1000 == 0:
            # Print loss / batch_size to match the C++ metric
            print(f"Step {step} | Loss: {(loss.item() / batch_size):.8f}")
            
            if step % 1000 == 0:
                out_name = f"../images/pt_pred_step_{step}.jpg"
                save_predicted_image(out_name, outputs.detach().float(), width, height)
                
    end_event.record()
    torch.cuda.synchronize()
    
    print("Training Completed. Saving final image...")
    save_predicted_image("../images/pt_pred_final.jpg", outputs.detach().float(), width, height)
    
    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / num_steps
    throughput = (batch_size * 1000.0) / avg_ms
    
    print("\n==============================================")
    print("           PERFORMANCE METRICS")
    print("==============================================")
    print(f"Avg Time per Step: {avg_ms:.6f} ms")
    print(f"Throughput:        {throughput:.5e} items/sec")
    print("==============================================")

if __name__ == "__main__":
    main()