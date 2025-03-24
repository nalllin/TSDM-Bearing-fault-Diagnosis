import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute sinusoidal embeddings for diffusion timesteps.
    """
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(1e4, device=device)) *
                      torch.arange(0, half_dim, device=device) / half_dim)
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm, convolution, and time-step embedding injection.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.activation = nn.SiLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        emb = self.time_proj(time_emb)
        h = h + emb[:, :, None]
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        return self.shortcut(x) + h

class AttentionBlock(nn.Module):
    """
    Self-attention block for 1D signals.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, L]
        B, C, L = x.size()
        h = x.permute(0, 2, 1)  # [B, L, C]
        h = self.norm(h)
        attn_out, _ = self.attn(h, h, h)
        h = attn_out + h
        h = h.permute(0, 2, 1)
        return h

class ImprovedUNet(nn.Module):
    """
    Improved U-Net for time-series diffusion with Residual and Attention Blocks.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 64, 
                 channel_mults=(1, 2, 4, 8), num_heads: int = 4, time_emb_dim: int = 256):
        super().__init__()
        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # Down-sampling path
        self.down_blocks = nn.ModuleList()
        self.attn_blocks_down = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = base_channels
        num_levels = len(channel_mults)
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(ResidualBlock(prev_ch, out_ch, time_emb_dim=time_emb_dim))
            if mult >= 4:
                self.attn_blocks_down.append(AttentionBlock(out_ch, num_heads=num_heads))
            else:
                self.attn_blocks_down.append(None)
            if i != num_levels - 1:
                self.downsamples.append(nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1))
            prev_ch = out_ch
        # Middle (bottleneck)
        self.mid_block1 = ResidualBlock(prev_ch, prev_ch, time_emb_dim=time_emb_dim)
        self.mid_attn = AttentionBlock(prev_ch, num_heads=num_heads)
        self.mid_block2 = ResidualBlock(prev_ch, prev_ch, time_emb_dim=time_emb_dim)
        # Up-sampling path
        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.attn_blocks_up = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults[:-1])):
            curr_ch = prev_ch
            skip_ch = base_channels * mult
            self.up_transposes.append(nn.ConvTranspose1d(curr_ch, skip_ch, kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(ResidualBlock(skip_ch * 2, skip_ch, time_emb_dim=time_emb_dim))
            if mult >= 4:
                self.attn_blocks_up.append(AttentionBlock(skip_ch, num_heads=num_heads))
            else:
                self.attn_blocks_up.append(None)
            prev_ch = skip_ch
        # Final output
        self.conv_out = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        # Embed time step and pass through MLP
        t_emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        h = self.conv_in(x)
        skip_connections = []
        for i, res_block in enumerate(self.down_blocks):
            h = res_block(h, t_emb)
            attn_block = self.attn_blocks_down[i]
            if attn_block is not None:
                h = attn_block(h)
            if i < len(self.downsamples):
                skip_connections.append(h)
                h = self.downsamples[i](h)
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        for j, up_conv in enumerate(self.up_transposes):
            skip_feat = skip_connections[-(j+1)]
            h = up_conv(h)
            # Adjust length if necessary
            if h.size(-1) != skip_feat.size(-1):
                diff = skip_feat.size(-1) - h.size(-1)
                if diff > 0:
                    h = F.pad(h, (0, diff))
                else:
                    h = h[:, :, :skip_feat.size(-1)]
            h = torch.cat([h, skip_feat], dim=1)
            h = self.up_blocks[j](h, t_emb)
            attn_block = self.attn_blocks_up[j]
            if attn_block is not None:
                h = attn_block(h)
        out = self.conv_out(h)
        return out
