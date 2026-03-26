import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load dataset from task1 (assume circuit1.py or circuit2.py provides X)
def load_data():
    # Placeholder: random data, replace with actual loading
    # Added Red Wine Quality dataset from UCI as an example in notebook
    X = np.random.rand(100, 10)  # 100 events, 10 features
    return X

# Simple diffusion model (DDPM-like, mock for demonstration)
# In notebook, implemented a proper diffusion process with noise scheduling and reverse sampling.
class DiffusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        return self.net(x)

# Training and evaluation

def train_diffusion():
    X = load_data()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model = DiffusionModel(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training
    for epoch in range(20):
        optimizer.zero_grad()
        recon = model(X_tensor)
        loss = criterion(recon, X_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Evaluation
    recon = model(X_tensor).detach().numpy()
    mse = mean_squared_error(X, recon)
    print(f"Diffusion Model MSE: {mse:.4f}")

    # Visual comparison
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(X, aspect='auto', cmap='viridis')
    plt.title('Original Events')
    plt.subplot(1,2,2)
    plt.imshow(recon, aspect='auto', cmap='viridis')
    plt.title('Reconstructed Events (Diffusion)')
    plt.tight_layout()
    plt.savefig('task10/diffusion_comparison.png')
    plt.show()
    # Save metric
    with open('task10/diffusion_results.txt', 'w') as f:
        f.write(f'Diffusion Model MSE: {mse:.4f}\n')

if __name__ == "__main__":
    train_diffusion()
