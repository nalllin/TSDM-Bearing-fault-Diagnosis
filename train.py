__all__ = ['train_model', 'sample']

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.models import ImprovedUNet
from models.dataset import BearingSignalDataset
import matplotlib.pyplot as plt
import os

# ---------------------------
# Configurable Diffusion Parameters
# ---------------------------
T = 3000  # Total number of diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1 - betas
alpha_cum = torch.cumprod(alphas, dim=0)
sqrt_alpha_cum = torch.sqrt(alpha_cum)
sqrt_one_minus_alpha_cum = torch.sqrt(1 - alpha_cum)

# ---------------------------
# Training Hyperparameters
# ---------------------------
num_epochs = 250
batch_size = 32
learning_rate = 1e-4

# ---------------------------
# Data Parameters
# ---------------------------
data_dir = r"C:\python_workspace\DDPM\raw"  # Folder with your .mat files
segment_length = 3000  # 0.25 s at 12,000 Hz
segment_step = 750     # Overlap step size

# ---------------------------
# Output Directory
# ---------------------------
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ---------------------------
# Forward Diffusion Function
# ---------------------------
def forward_diffusion(x0: torch.Tensor, t: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Compute x_t from x0 at diffusion step t:
        x_t = sqrt(alpha_cum[t]) * x0 + sqrt(1 - alpha_cum[t]) * noise
    """
    t_cpu = t.cpu()
    sqrt_ac = sqrt_alpha_cum[t_cpu].view(-1, 1, 1).to(x0.device)
    sqrt_omc = sqrt_one_minus_alpha_cum[t_cpu].view(-1, 1, 1).to(x0.device)
    noise = torch.randn_like(x0)
    x_t = sqrt_ac * x0 + sqrt_omc * noise
    return x_t, noise

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, dataloader, num_epochs, device):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x0, _ in dataloader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            x_t, noise = forward_diffusion(x0, t)
            noise_pred = model(x_t, t)
            loss = mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Plot epoch vs. loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Epoch vs. Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epoch_vs_loss.png"))
    plt.show()
    
    return epoch_losses

# ---------------------------
# Sampling Function (Reverse Diffusion)
# ---------------------------
def sample(model, num_samples=1, device="cpu"):
    """
    Generate synthetic time-series samples using the reverse diffusion process.
    Returns a list of generated signals.
    """
    model.eval()
    generated_signals = []
    for i in range(num_samples):
        # Start from x_T ~ N(0, I)
        x = torch.randn(1, 1, segment_length).to(device)
        # Reverse diffusion: t = T-1, T-2, ..., 0
        for t in range(T-1, -1, -1):
            t_tensor = torch.tensor([t], device=device)
            noise_pred = model(x, t_tensor)
            sqrt_ac = torch.sqrt(alpha_cum[t]).to(device)
            sqrt_omc = torch.sqrt(1 - alpha_cum[t]).to(device)
            # Estimate x0 from x_t
            x0_pred = (x - sqrt_omc * noise_pred) / sqrt_ac
            if t > 0:
                z = torch.randn_like(x)
                sqrt_alpha_cum_prev = torch.sqrt(alpha_cum[t-1]).to(device)
                sqrt_omc_prev = torch.sqrt(1 - alpha_cum[t-1]).to(device)
                # Compute x_{t-1} from predicted x0
                x = sqrt_alpha_cum_prev * x0_pred + sqrt_omc_prev * z
            else:
                x = x0_pred
        generated_signals.append(x.squeeze().detach().cpu().numpy())
    return generated_signals

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    from torch.utils.data import DataLoader
    dataset = BearingSignalDataset(data_dir=data_dir, segment_length=segment_length, segment_step=segment_step)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = ImprovedUNet(in_channels=1, base_channels=64, channel_mults=(1, 2, 4, 8), time_emb_dim=256)
    
    # Train the model
    loss_history = train_model(model, dataloader, num_epochs=num_epochs, device=device)
    
    # Save trained model and loss history
    torch.save(model.state_dict(), os.path.join(output_dir, "trained_model.pth"))
    with open(os.path.join(output_dir, "loss_history.txt"), "w") as f:
        for epoch, loss in enumerate(loss_history, 1):
            f.write(f"Epoch {epoch}: {loss:.4f}\n")
