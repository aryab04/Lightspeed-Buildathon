import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ddpm import DDPM, DDPMScheduler, SimpleUNet
from torch.utils.data import DataLoader, TensorDataset


class Client:
    def __init__(self, client_id, data_loader, local_time_steps, device="cuda"):
        self.client_id = client_id
        self.data_loader = data_loader
        self.t0 = local_time_steps
        self.device = device
        
        # Initialize local model (personalized denoiser)
        self.model = SimpleUNet().to(device)
        self.scheduler = DDPMScheduler(num_timesteps=local_time_steps)
        self.ddpm = DDPM(self.model, self.scheduler, device)
        
    def train_local_model(self, num_epochs, optimizer):
        """Train a personalized secret denoiser (line 2 in Algorithm 3)"""
        for epoch in range(num_epochs):
            running_loss = 0.0
            num_batches = 0
            
            for batch in self.data_loader:
                x_0 = batch[0].to(self.device)
                loss = self.ddpm.train_step(x_0, optimizer)
                running_loss += loss
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = running_loss / num_batches
                print(f"Client {self.client_id}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
    def create_noisy_dataset(self, num_samples):
        """Create a noisy dataset (lines 3-8 in Algorithm 3)"""
        batch_size = next(iter(self.data_loader))[0].shape[0]
        sample_batches = (num_samples + batch_size - 1) // batch_size
        noisy_samples = []
        
        for _ in range(sample_batches):
            # Sample real data
            for batch in self.data_loader:
                x_0 = batch[0].to(self.device)
                
                # Set αt = Πts=1(1 − βs)
                alpha_t = torch.prod(1 - self.scheduler.betas[:self.t0])
                
                # Sample noise
                z = torch.randn_like(x_0)
                
                # Obtain noisy data: x_0^m = sqrt(α_t)x_0^m + sqrt(1-α_t)z
                noisy_data = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * z
                
                noisy_samples.append(noisy_data.cpu())
                break
                
        return torch.cat(noisy_samples[:num_samples], dim=0)


class Server:
    def __init__(self, global_time_steps, device="cuda"):
        self.T = global_time_steps
        self.device = device
        
        # Initialize global model (shared denoiser)
        self.model = SimpleUNet().to(device)
        self.scheduler = DDPMScheduler(num_timesteps=global_time_steps)
        self.ddpm = DDPM(self.model, self.scheduler, device)
        
    def train_global_model(self, noisy_datasets, num_epochs, optimizer):
        """Train a shared global denoiser (lines 11-14 in Algorithm 3)"""
        # Combine noisy datasets from all clients
        combined_data = torch.cat(noisy_datasets, dim=0)
        dataset = TensorDataset(combined_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                x_0 = batch[0].to(self.device)
                loss = self.ddpm.train_step(x_0, optimizer)
                running_loss += loss
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = running_loss / num_batches
                print(f"Server, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")


def personalized_federated_training(clients, server, client_epochs, server_epochs, client_optimizer_fn, server_optimizer_fn):
    """Personalized Federated Training of Diffusion Model (Algorithm 3)"""
    # Stage 1: Client-side training
    print("Starting client-side training...")
    for client in clients:
        optimizer = client_optimizer_fn(client.model.parameters())
        client.train_local_model(client_epochs, optimizer)
        
        # Create noisy dataset
        client.noisy_dataset = client.create_noisy_dataset(num_samples=100)
        print(f"Client {client.client_id} created noisy dataset with shape {client.noisy_dataset.shape}")
    
    # Stage 2: Server-side training
    print("Starting server-side training...")
    noisy_datasets = [client.noisy_dataset for client in clients]
    optimizer = server_optimizer_fn(server.model.parameters())
    server.train_global_model(noisy_datasets, server_epochs, optimizer)
    
    return server, clients


def sampling_procedure(server, client, num_samples, image_size, channels=1):
    """Sampling Procedure for PFDM (Algorithm 4)"""
    # Start with shared global denoiser to generate x̂_0
    x_0 = server.ddpm.sample(num_samples, image_size, channels)
    
    # Set αt = Πts=1(1 − βs)
    alpha_t = torch.prod(1 - client.scheduler.betas)
    
    # Initialize x_t for reverse process
    for t in range(client.t0, 0, -1):
        t_tensor = torch.full((num_samples,), t-1, device=client.device, dtype=torch.long)
        
        # Get model prediction
        noise_pred = client.model(x_0, t_tensor)
        
        # Calculate parameters
        beta_t = client.scheduler.betas[t-1]
        
        # Sample noise
        z = torch.randn_like(x_0) if t > 1 else torch.zeros_like(x_0)
        
        # Equation 5 in Algorithm 4
        x_t_minus_1 = (1 / torch.sqrt(1 - beta_t)) * (
            x_0 - (beta_t / torch.sqrt(1 - alpha_t)) * client.model(x_0, t_tensor)
        ) + torch.sqrt(beta_t) * z
        
        x_0 = x_t_minus_1
        
    return x_0
