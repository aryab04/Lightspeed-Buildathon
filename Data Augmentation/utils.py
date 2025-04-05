import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def visualize_samples(samples, nrow=8):
    """Visualize a batch of samples"""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    
    # Make grid
    grid = make_grid(samples, nrow=nrow)
    
    # Convert to numpy and transpose
    grid = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def calculate_alpha_t(betas, t):
    """Calculate alpha_t = Πts=1(1-βs)"""
    return torch.prod(1 - betas[:t])


def evaluate_fid(real_samples, generated_samples):
    """
    Placeholder for FID score calculation
    
    Note: Actual FID calculation would require additional dependencies
    """
    # This is a placeholder - actual implementation would use libraries
    # like torch-fidelity for proper FID calculation
    print("FID score calculation would go here")
    return 0.0
