

import torch
from torch.utils.data import TensorDataset, DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Known matrices
A_true = torch.tensor([[1.0, 0.1], [0.0, 1.0]], dtype=torch.float32, device=device)
B_true = torch.tensor([[0.0], [0.5]], dtype=torch.float32, device=device)

# Number of simulation steps
n_steps = 200

# Initialize state and input
x_data = []
u_data = []
x_k = torch.tensor([[0.0], [0.0]], dtype=torch.float32, device=device)  # Initial state as a column vector (2x1)

# Generate random inputs
u_k_values = torch.randn(n_steps, 1, dtype=torch.float32, device=device)  # Inputs as column vectors (nx1)

for k in range(n_steps):
    # Ensure x_k is a 2x1 column vector
    x_data.append(x_k)
    u_data.append(u_k_values[k].view(-1, 1))

    # Compute next state using true A and B
    x_k_1 = A_true @ x_data[k] + B_true @ u_data[k]

    # Update the state
    x_k = x_k_1

# Convert data to tensors (stacking column vectors)
x_data = torch.stack(x_data, dim=0)  # Shape: (n_steps, 2, 1)
u_data = torch.stack(u_data, dim=0)  # Shape: (n_steps, 1)

# Prepare input (x_k, u_k) and target (x_{k+1}) tensors
x_k_data = x_data[:-1]  # All states except the last one
u_k_data = u_data[:-1]  # All inputs except the last one
x_k_1_targets = x_data[1:]  # True next states

# Create a dataset and data loader
dataset = TensorDataset(x_k_data, u_k_data, x_k_1_targets)
batch_size = 16  # Define the batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


print(f"x_data shape: {x_data.shape}")
print(f"u_data shape: {u_data.shape}")

# Model definition
class LinearDynamicModel(torch.nn.Module):
    def __init__(self):
        super(LinearDynamicModel, self).__init__()
        # Initialize A and B as learnable parameters
        self.A = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float32, device=device) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(2, 1, dtype=torch.float32, device=device) * 0.1)
    
    def forward(self, x_k, u_k):
        # Predict next state x_{k+1}
        return self.A @ x_k + self.B @ u_k



# Instantiate the model and optimizer
model = LinearDynamicModel().to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Training parameters
n_epochs = 200

# Training loop
for epoch in range(n_epochs):
    total_loss = 0.0
    
    # Iterate over batches
    for x_k_batch, u_k_batch, x_k_1_batch in data_loader:
        # Move batch data to the device
        x_k_batch = x_k_batch.to(device)
        u_k_batch = u_k_batch.to(device)
        x_k_1_batch = x_k_1_batch.to(device)
        
        # Predicted next state for the batch
        x_k_1_pred = model.forward(x_k_batch, u_k_batch)
        
        # Compute loss for the batch
        loss = loss_fn(x_k_1_pred, x_k_1_batch)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_k_batch.size(0)  # Accumulate total loss (scaled by batch size)
    
    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        average_loss = total_loss / len(dataset)
        print(f'Epoch {epoch + 1}/{n_epochs}, Average Loss: {average_loss:.6f}')

# Print the learned parameters
print("\nLearned parameters:")
# Set print precision to 3 decimal places
print("A learned:\n", model.A.data)
print("B learned:\n", model.B.data)