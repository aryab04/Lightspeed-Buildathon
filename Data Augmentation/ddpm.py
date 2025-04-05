import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Define beta schedule as in equation
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Pre-calculate different terms
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def compute_alpha_t(self, t):
        """Compute the cumulative product of (1 - beta) up to index t"""
        return self.alphas_cumprod[t]


class SimpleUNet(nn.Module):
    """A simplified U-Net architecture for denoising"""
    def __init__(self, channels=1, time_emb_dim=32):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        
        # Time embedding projections
        self.temb_proj1 = nn.Linear(time_emb_dim, 64)
        self.temb_proj2 = nn.Linear(time_emb_dim, 128)
        self.temb_proj3 = nn.Linear(time_emb_dim, 256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)  # Skip connection doubles channels
        self.final = nn.Conv2d(128, channels, 3, padding=1)  # Skip connection doubles channels
        
    def forward(self, x, t):
        # Embed time
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Encoder
        h1 = F.silu(self.conv1(x))
        h1 = h1 + self.temb_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        h2 = F.silu(self.conv2(h1))
        h2 = h2 + self.temb_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        h3 = F.silu(self.conv3(h2))
        h3 = h3 + self.temb_proj3(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Decoder with skip connections
        h = F.silu(self.up1(h3))
        h = torch.cat([h, h2], dim=1)
        
        h = F.silu(self.up2(h))
        h = torch.cat([h, h1], dim=1)
        
        return self.final(h)


class DDPM:
    def __init__(self, denoiser, scheduler, device="cuda"):
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.device = device
        self.denoiser.to(device)
        
    def train_step(self, x_0, optimizer):
        """Single training step for DDPM (Algorithm 1)"""
        self.denoiser.train()
        batch_size = x_0.shape[0]
        
        # Sample t ~ Uniform({1, ..., T})
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device).long()
        
        # Sample noise to add to the images
        noise = torch.randn_like(x_0)
        
        # Diffuse the images
        alpha_t = self.scheduler.compute_alpha_t(t)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        # Noisy images: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * z where z ~ N(0, I)
        noisy_images = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        
        # Predict the noise using the model
        noise_pred = self.denoiser(noisy_images, t)
        
        # Calculate loss (MSE between predicted and actual noise)
        loss = F.mse_loss(noise_pred, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, num_samples, image_size, channels=1):
        """Sampling procedure for DDPM (Algorithm 2)"""
        self.denoiser.eval()
        
        # Start from pure noise
        x = torch.randn(num_samples, channels, *image_size, device=self.device)
        
        # Iterate from T to 1
        for t in range(self.scheduler.num_timesteps - 1, -1, -1):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.denoiser(x, t_tensor)
            
            # Calculate parameters
            alpha_t = self.scheduler.alphas_cumprod[t]
            alpha_t_prev = self.scheduler.alphas_cumprod_prev[t]
            beta_t = self.scheduler.betas[t]
            
            # No noise at timestep 0
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            
            # Equation 5 in Algorithm 2
            x_t_minus_1 = (1 / torch.sqrt(1 - beta_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise
            
            x = x_t_minus_1
            
        return x
