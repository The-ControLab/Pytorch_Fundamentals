"""
===============================================================================
Module Name:    ex_torch_linear_regression.py
Description:    This module trains a one variable linear regression y = a*x + b
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

# ========================================
# Create the training data set
# ========================================

# Model Inputs
X = torch.arange(0, 10, dtype=torch.float32, device=device)  # Use float for better accuracy

# Compute True Outputs
a_true = 2.0
b_true = 1.0
Y = a_true * X + b_true

# ========================================
# Initialize Training Parameters
# ========================================
# Random initialization with small values
a = torch.nn.Parameter(torch.randn(1, device=device) * 0.01)
b = torch.nn.Parameter(torch.randn(1, device=device) * 0.01)

# ========================================
# Define Model for Training
# ========================================

def forward(X):
    """Compute the model output given input X."""
    return a * X + b

# Loss = MSE
loss_fn = torch.nn.MSELoss()

print(f'Prediction before training: f(5) = {forward(torch.tensor(5.0, device=device)).item():.3f}')

# ========================================
# Training
# ========================================
learning_rate = 0.01
n_iters = 1200

# Optimizer setup
optimizer = torch.optim.RMSprop([a, b], lr=learning_rate)

for epoch in range(n_iters):
    # Predict (forward pass)
    Yhat = forward(X)

    # Compute the loss
    l = loss_fn(Y, Yhat)

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
