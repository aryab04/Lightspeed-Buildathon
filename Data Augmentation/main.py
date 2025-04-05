import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os

from ddpm import DDPM, DDPMScheduler, SimpleUNet
from pfdm import Client, Server, personalized_federated_training, sampling_procedure
from privacy import compute_privacy_guarantee


def load_mnist(batch_size=64, split_into=3, subset_size=3000):
    """Load MNIST dataset and split into multiple clients"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # REDUCE DATASET SIZE: Take only a small subset instead of full 60,000
    indices = torch.randperm(len(dataset))[:subset_size]
    dataset = torch.utils.data.Subset(dataset, indices)
    
    # Split dataset for clients
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    split_size = dataset_size // split_into
    client_indices = [indices[i*split_size:(i+1)*split_size] for i in range(split_into)]
    
    # Create dataloaders
    client_dataloaders = []
    for indices in client_indices:
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(dataloader)
    
    return client_dataloaders


def save_samples(samples, path):
    """Save generated samples as an image grid"""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    
    # Create grid of images
    grid_size = int(np.sqrt(samples.shape[0]))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    for i, ax in enumerate(axs.flatten()):
        if i < samples.shape[0]:
            if samples.shape[1] == 1:  # Grayscale
                ax.imshow(samples[i, 0].detach().cpu().numpy(), cmap='gray')
            else:  # RGB
                ax.imshow(samples[i].permute(1, 2, 0).detach().cpu().numpy())
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters - OPTIMIZED FOR HIGH-QUALITY OUTPUTS
    num_clients = 3             # More clients for better collaborative learning
    local_time_steps = 200      # Increased for better personalization
    global_time_steps = 800     # Maintain total of 1000 steps
    client_epochs = 8           # Increased from 2 to 8 for better local model convergence
    server_epochs = 15          # Increased from 3 to 15 for better global model convergence
    batch_size = 128            # Increased for more stable gradient updates
    learning_rate = 1e-4        # Slightly lower for more stable convergence
    image_size = (28, 28)       # MNIST
    channels = 1
    
    # Load reduced datasets - still using reduced size for practical runtime
    client_dataloaders = load_mnist(batch_size, num_clients, subset_size=3000)
    
    print(f"Using reduced MNIST dataset with 3000 samples split across {num_clients} clients")
    print(f"Optimized parameters: t0={local_time_steps}, T-t0={global_time_steps}, client_epochs={client_epochs}, server_epochs={server_epochs}")
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Initialize clients
    clients = []
    for i in range(num_clients):
        clients.append(Client(
            client_id=i,
            data_loader=client_dataloaders[i],
            local_time_steps=local_time_steps,
            device=device
        ))
    
    # Initialize server
    server = Server(
        global_time_steps=global_time_steps,
        device=device
    )
    
    # Define optimizer factories with gradient clipping for stability
    def client_optimizer_fn(params):
        optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    
    def server_optimizer_fn(params):
        optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    
    # Train PFDM
    server, clients = personalized_federated_training(
        clients,
        server,
        client_epochs,
        server_epochs,
        client_optimizer_fn,
        server_optimizer_fn
    )
    
    # Calculate privacy guarantees
    beta_schedule = (1e-4, 0.02)  # As defined in the paper
    epsilon = compute_privacy_guarantee(
        t0=local_time_steps,
        T=global_time_steps + local_time_steps,
        beta_schedule=beta_schedule
    )
    print(f"Privacy guarantee: (ε={epsilon:.2f}, δ=1e-5)-DP")
    
    # Generate samples for each client
    num_samples = 16 
    for i, client in enumerate(clients):
        samples = sampling_procedure(
            server,
            client,
            num_samples=num_samples,
            image_size=image_size,
            channels=channels
        )
        
        # Save samples
        save_samples(samples, f"results/client_{i}_samples.png")
        print(f"Generated samples for client {i}")


if __name__ == "__main__":
    main()




#-----------------------------------------------------------



# import torch
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import datasets, transforms
# import os

# from ddpm import DDPM, DDPMScheduler, SimpleUNet
# from pfdm import Client, Server, personalized_federated_training, sampling_procedure
# from privacy import compute_privacy_guarantee


# def load_mnist(batch_size=64, split_into=3, subset_size=3000):
#     """Load MNIST dataset and split into multiple clients"""
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
    
#     dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
#     # REDUCE DATASET SIZE: Take only a small subset instead of full 60,000
#     indices = torch.randperm(len(dataset))[:subset_size]
#     dataset = torch.utils.data.Subset(dataset, indices)
    
#     # Split dataset for clients
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     np.random.shuffle(indices)
    
#     split_size = dataset_size // split_into
#     client_indices = [indices[i*split_size:(i+1)*split_size] for i in range(split_into)]
    
#     # Create dataloaders
#     client_dataloaders = []
#     for indices in client_indices:
#         subset = torch.utils.data.Subset(dataset, indices)
#         dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
#         client_dataloaders.append(dataloader)
    
#     return client_dataloaders


# def save_samples(samples, path):
#     """Save generated samples as an image grid"""
#     # Denormalize from [-1, 1] to [0, 1]
#     samples = (samples + 1) / 2
    
#     # Create grid of images
#     grid_size = int(np.sqrt(samples.shape[0]))
#     fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
#     for i, ax in enumerate(axs.flatten()):
#         if i < samples.shape[0]:
#             if samples.shape[1] == 1:  # Grayscale
#                 ax.imshow(samples[i, 0].detach().cpu().numpy(), cmap='gray')
#             else:  # RGB
#                 ax.imshow(samples[i].permute(1, 2, 0).detach().cpu().numpy())
#             ax.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(path)
#     plt.close()


# def main():
#     # Set random seed for reproducibility
#     torch.manual_seed(42)
#     np.random.seed(42)
    
#     # Use GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Parameters - REDUCED FOR FASTER TRAINING
#     num_clients = 2        # Reduced from 3 to 2
#     local_time_steps = 50  # Reduced from 100 to 50
#     global_time_steps = 450 # Reduced from 900 to 450
#     client_epochs = 2      # Reduced from 5 to 2
#     server_epochs = 3      # Reduced from 10 to 3
#     batch_size = 32        # Reduced from 64 to 32
#     learning_rate = 2e-4
#     image_size = (28, 28)  # MNIST
#     channels = 1
    
#     # Load reduced datasets
#     client_dataloaders = load_mnist(batch_size, num_clients, subset_size=3000)
    
#     print(f"Using reduced MNIST dataset with 3000 samples split across {num_clients} clients")
    
#     # Create output directory
#     os.makedirs("results", exist_ok=True)
    
#     # Initialize clients
#     clients = []
#     for i in range(num_clients):
#         clients.append(Client(
#             client_id=i,
#             data_loader=client_dataloaders[i],
#             local_time_steps=local_time_steps,
#             device=device
#         ))
    
#     # Initialize server
#     server = Server(
#         global_time_steps=global_time_steps,
#         device=device
#     )
    
#     # Define optimizer factories
#     def client_optimizer_fn(params):
#         return optim.Adam(params, lr=learning_rate)
    
#     def server_optimizer_fn(params):
#         return optim.Adam(params, lr=learning_rate)
    
#     # Train PFDM
#     server, clients = personalized_federated_training(
#         clients,
#         server,
#         client_epochs,
#         server_epochs,
#         client_optimizer_fn,
#         server_optimizer_fn
#     )
    
#     # Calculate privacy guarantees
#     beta_schedule = (1e-4, 0.02)  # As defined in the paper
#     epsilon = compute_privacy_guarantee(
#         t0=local_time_steps,
#         T=global_time_steps + local_time_steps,
#         beta_schedule=beta_schedule
#     )
#     print(f"Privacy guarantee: (ε={epsilon:.2f}, δ=1e-5)-DP")
    
#     # Generate samples for each client - reduced sample count
#     num_samples = 16 
#     for i, client in enumerate(clients):
#         samples = sampling_procedure(
#             server,
#             client,
#             num_samples=num_samples,
#             image_size=image_size,
#             channels=channels
#         )
        
#         # Save samples
#         save_samples(samples, f"results/client_{i}_samples.png")
#         print(f"Generated samples for client {i}")


# if __name__ == "__main__":
#     main()















#----------------------------------------------------------

# import torch
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import datasets, transforms
# import os

# from ddpm import DDPM, DDPMScheduler, SimpleUNet
# from pfdm import Client, Server, personalized_federated_training, sampling_procedure
# from privacy import compute_privacy_guarantee


# def load_mnist(batch_size=64, split_into=3):
#     """Load MNIST dataset and split into multiple clients"""
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
    
#     dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
#     # Split dataset for clients
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     np.random.shuffle(indices)
    
#     split_size = dataset_size // split_into
#     client_indices = [indices[i*split_size:(i+1)*split_size] for i in range(split_into)]
    
#     # Create dataloaders
#     client_dataloaders = []
#     for indices in client_indices:
#         subset = torch.utils.data.Subset(dataset, indices)
#         dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
#         client_dataloaders.append(dataloader)
    
#     return client_dataloaders


# def save_samples(samples, path):
#     """Save generated samples as an image grid"""
#     # Denormalize from [-1, 1] to [0, 1]
#     samples = (samples + 1) / 2
    
#     # Create grid of images
#     grid_size = int(np.sqrt(samples.shape[0]))
#     fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
#     for i, ax in enumerate(axs.flatten()):
#         if i < samples.shape[0]:
#             if samples.shape[1] == 1:  # Grayscale
#                 ax.imshow(samples[i, 0].detach().cpu().numpy(), cmap='gray')
#             else:  # RGB
#                 ax.imshow(samples[i].permute(1, 2, 0).detach().cpu().numpy())
#             ax.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(path)
#     plt.close()


# def main():
#     # Set random seed for reproducibility
#     torch.manual_seed(42)
#     np.random.seed(42)
    
#     # Use GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Parameters
#     num_clients = 3
#     local_time_steps = 100  # t0
#     global_time_steps = 900  # T - t0
#     client_epochs = 5
#     server_epochs = 10
#     batch_size = 64
#     learning_rate = 2e-4
#     image_size = (28, 28)  # MNIST
#     channels = 1
    
#     # Load datasets
#     client_dataloaders = load_mnist(batch_size, num_clients)
    
#     # Create output directory
#     os.makedirs("results", exist_ok=True)
    
#     # Initialize clients
#     clients = []
#     for i in range(num_clients):
#         clients.append(Client(
#             client_id=i,
#             data_loader=client_dataloaders[i],
#             local_time_steps=local_time_steps,
#             device=device
#         ))
    
#     # Initialize server
#     server = Server(
#         global_time_steps=global_time_steps,
#         device=device
#     )
    
#     # Define optimizer factories
#     def client_optimizer_fn(params):
#         return optim.Adam(params, lr=learning_rate)
    
#     def server_optimizer_fn(params):
#         return optim.Adam(params, lr=learning_rate)
    
#     # Train PFDM
#     server, clients = personalized_federated_training(
#         clients,
#         server,
#         client_epochs,
#         server_epochs,
#         client_optimizer_fn,
#         server_optimizer_fn
#     )
    
#     # Calculate privacy guarantees
#     beta_schedule = (1e-4, 0.02)  # As defined in the paper
#     epsilon = compute_privacy_guarantee(
#         t0=local_time_steps,
#         T=global_time_steps + local_time_steps,
#         beta_schedule=beta_schedule
#     )
#     print(f"Privacy guarantee: (ε={epsilon:.2f}, δ=1e-5)-DP")
    
#     # Generate samples for each client
#     num_samples = 16
#     for i, client in enumerate(clients):
#         samples = sampling_procedure(
#             server,
#             client,
#             num_samples=num_samples,
#             image_size=image_size,
#             channels=channels
#         )
        
#         # Save samples
#         save_samples(samples, f"results/client_{i}_samples.png")
#         print(f"Generated samples for client {i}")


# if __name__ == "__main__":
#     main()
