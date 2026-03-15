import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cirq
import matplotlib.pyplot as plt

# Generate normally distributed data
N = 200
input_dim = 8
X = np.random.normal(0, 1, size=(N, input_dim))
Y = np.random.normal(0, 1, size=(N, 5))  # Target PQC parameters (for demonstration)

# MLP to estimate PQC parameters
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# PQC: 5 qubits, RX rotations

def pqc_state(params):
    qubits = [cirq.GridQubit(0, i) for i in range(5)]
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(params[i])(qubit))
    return circuit, qubits

# Simple DQN for temporal difference learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Training

def train_td():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    model = DQN(input_dim, 5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    gamma = 0.99
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        # Temporal difference target: Y + gamma * pred (mock, since no environment)
        td_target = Y_tensor + gamma * pred.detach()
        loss = criterion(pred, td_target)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, TD Loss: {loss.item():.4f}")

    # Visualize PQC for one sample
    params = model(X_tensor[:1]).detach().cpu().numpy()[0]
    circuit, qubits = pqc_state(params)
    print("Sample PQC circuit:")
    print(circuit)
    # Save circuit diagram
    from cirq.contrib.svg import circuit_to_svg
    svg = circuit_to_svg(circuit)
    with open("task12/pqc_td_sample.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    print("Sample PQC circuit saved as task12/pqc_td_sample.svg")

if __name__ == "__main__":
    train_td()
