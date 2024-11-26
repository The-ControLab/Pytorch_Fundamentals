"""
===============================================================================
Module Name:    ex_torch_pendulum_sysID.py
Description:    This module searchs the parameters of the linear inverted pendulum
                model using PyTorch.
Author:         The ControLab
Date:           2024-10-26
Version:        1.1
===============================================================================
"""

import torch

# ========================================
# Device Setup (use GPU if available)
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ========================================
# Create Training Parameters
# ========================================
# Random initialization with small values
mp = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))
mc = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32, device=device))
l = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=device))
k = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32, device=device))
bp = torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float32, device=device))
bc = torch.nn.Parameter(torch.tensor(0.002, dtype=torch.float32, device=device))
g = 9.81


# ========================================
# Define non-linear State Space Model
# ========================================
# Sampling Frequency
Ts = 1/1000


def foward_x_k_1(x, x_dot, theta, theta_dot, u, Ts):
    x_k_1 = x + x_dot*Ts
    theta_k_1 = theta + theta_dot*Ts

    num = mp*torch.cos(theta)*(bp*theta_dot - g*torch.sin(theta)) + k*u - bc*x_dot + mp*l*theta_dot*theta_dot*torch.sin(theta)
    den = mc + mp*(1+torch.cos(theta)*torch.cos(theta))
    x_dot_k_1 = x_dot + Ts*num/den

    theta_dot_k_1 = (-bp*theta_dot - x_dot_k_1*torch.cos(theta) + g*torch.sin(theta))/l

    return [x_k_1, theta_k_1, x_dot_k_1, theta_dot_k_1]


# ========================================
# Create the training data set
# ========================================

# Control Input
X = torch.arange(0, 10, dtype=torch.float32, device=device)  # Use float for better accuracy

# Compute True Outputs Simulating the System
a_true = 2.0
b_true = 1.0
Y = a_true * X + b_true



# ========================================
# Define Model for Training
# ========================================

def forward(X):
    """Compute the model output given input X."""
    return a * X + b

# Loss = MSE
loss = torch.nn.MSELoss()

print(f'Prediction before training: f(5) = {forward(torch.tensor(5.0, device=device)).item():.3f}')

# ========================================
# Training
# ========================================
learning_rate = 0.01
n_iters = 1200

# Random initialization with small values
a = torch.nn.Parameter(torch.randn(1, device=device) * 0.01)
b = torch.nn.Parameter(torch.randn(1, device=device) * 0.01)

# Optimizer setup
optimizer = torch.optim.RMSprop([a, b], lr=learning_rate)

for epoch in range(n_iters):
    # Predict (forward pass)
    Yhat = forward(X)

    # Compute the loss
    l = loss(Y, Yhat)

    # Backward pass (compute gradients)
    l.backward()

    # Update parameters (using gradients)
    optimizer.step()

    # Zero the gradients after updating
    optimizer.zero_grad()

    # Logging progress
    if (epoch + 1) % 200 == 0:
        print(f'Epoch {epoch + 1}: a = {a.item():.3f}, b = {b.item():.3f}, Loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {forward(torch.tensor(5.0, device=device)).item():.3f}')
