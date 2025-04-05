import torch
import numpy as np


def compute_privacy_guarantee(t0, T, beta_schedule, C=1.0, delta=1e-5):
    """
    Compute the differential privacy guarantee for PFDM (Theorem 5.1)
    
    Args:
        t0: Local time steps
        T: Global time steps
        beta_schedule: Tuple (beta_start, beta_end)
        C: Norm bound (default: 1.0)
        delta: Delta parameter for (epsilon, delta)-DP (default: 1e-5)
        
    Returns:
        epsilon: Privacy parameter for (epsilon, delta)-DP
    """
    # Generate beta schedule
    beta_start, beta_end = beta_schedule
    betas = np.linspace(beta_start, beta_end, T)
    
    # Calculate alphas and alpha_t0
    alphas = 1 - betas
    alpha_t0 = np.prod(alphas[:t0])
    
    # Calculate epsilon according to Theorem 5.1
    epsilon_term1 = (2 * alpha_t0 * C**2) / (1 - alpha_t0)
    epsilon_term2 = np.sqrt((8 * t0 * np.log(1/delta)) / (1 - alpha_t0))
    
    epsilon = epsilon_term1 + epsilon_term2
    
    return epsilon


class LaplaceMechanism:
    """Implementation of the Laplace Mechanism for differential privacy"""
    def __init__(self, epsilon, sensitivity=1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
    def add_noise(self, tensor):
        """Add Laplace noise to tensor"""
        scale = self.sensitivity / self.epsilon
        noise = torch.from_numpy(
            np.random.laplace(0, scale, size=tensor.shape)
        ).to(tensor.device).to(tensor.dtype)
        
        return tensor + noise
