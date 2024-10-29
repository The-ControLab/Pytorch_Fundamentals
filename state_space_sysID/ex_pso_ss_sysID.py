import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Known matrices (True values for generating data)
A_true = torch.tensor([[1.0, 0.1],
                       [0.0, 1.0]], dtype=torch.float32, device=device)
B_true = torch.tensor([[0.0],
                       [0.5]], dtype=torch.float32, device=device)

# Number of simulation steps
n_steps = 100

# Initialize state and input
x_data = []
u_data = []
x_k = torch.tensor([[0.0],
                    [0.0]], dtype=torch.float32, device=device)  # Initial state

# Append the initial state
x_data.append(x_k.clone())

# Generate random inputs
u_k_values = torch.randn(n_steps, 1, dtype=torch.float32, device=device)

# Generate the data using the true A and B matrices
for k in range(n_steps):
    u_k = u_k_values[k].view(1, 1)
    u_data.append(u_k)

    # Compute next state
    x_k = A_true @ x_k + B_true @ u_k.view(-1, 1)

    # Append the new state
    x_data.append(x_k.clone())

# Stack to form tensors
x_data = torch.stack(x_data, dim=0)  # Shape: (n_steps + 1, 2, 1)
u_data = torch.stack(u_data, dim=0)   # Shape: (n_steps, 1, 1)

# ==========================================
# Normalization of Data
# ==========================================
x_mean = x_data.mean(dim=0)
x_std = x_data.std(dim=0)
u_mean = u_data.mean(dim=0)
u_std = u_data.std(dim=0)

# Normalize data
x_data_norm = (x_data - x_mean) / x_std
u_data_norm = (u_data - u_mean) / u_std

# ==========================================
# Define the State-Space Model Function
# ==========================================
def simulate_state_space(A, B, x0, u_data):
    x_pred = []
    x_k = x0

    # Simulate and store predictions
    for u_k in u_data:
        x_k = A @ x_k + B @ u_k.view(-1, 1)
        x_pred.append(x_k)

    x_pred = torch.stack(x_pred, dim=0)  # Shape: (n_steps, 2, 1)
    return x_pred

# ==========================================
# Define the Objective Function
# ==========================================
def objective_function(A_norm, B_norm, x_data, u_data):
    # Denormalize A and B
    A, B = denormalize_params(A_norm, B_norm)

    # Initial state for simulation (normalized)
    x0 = x_data[0]
    x_pred = simulate_state_space(A, B, x0, u_data)

    # Compute the mean squared error
    return torch.nn.functional.mse_loss(x_pred, x_data[1:])  # Both have length n_steps

# ==========================================
# Denormalize Parameters Function
# ==========================================
def denormalize_params(A_norm, B_norm):
    # Adjust this function based on your normalization approach
    return A_norm, B_norm

# ==========================================
# PSO Initialization and Training
# ==========================================
n_particles = 200
n_iters = 150
inertia_weight = 0.7
c1, c2 = 1.5, 1.5

# Initialize particles' positions and velocities for normalized parameters
particle_positions = torch.rand((n_particles, 6), dtype=torch.float32, device=device) * 0.2 - 0.1
particle_velocities = torch.rand((n_particles, 6), dtype=torch.float32, device=device) * 0.02 - 0.01

personal_best_positions = particle_positions.clone()
personal_best_scores = torch.full((n_particles,), float('inf'), device=device)
global_best_position = personal_best_positions[0]
global_best_score = float('inf')

# Reshape normalized parameters into A and B
def reshape_params(particle_position):
    A = particle_position[:4].view(2, 2)
    B = particle_position[4:].view(2, 1)
    return A, B

# Run PSO on normalized data
for iter in range(n_iters):
    for i in range(n_particles):
        # Reshape the particle's position into A_norm and B_norm
        A_norm, B_norm = reshape_params(particle_positions[i])

        # Calculate the objective function (error) for the particle
        score = objective_function(A_norm, B_norm, x_data_norm, u_data_norm)

        # Update personal best
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = particle_positions[i].clone()

        # Update global best
        if score < global_best_score:
            global_best_score = score
            global_best_position = particle_positions[i].clone()

    # Update particle velocities and positions
    r1, r2 = torch.rand(2, n_particles, 6, device=device)
    cognitive_velocity = c1 * r1 * (personal_best_positions - particle_positions)
    social_velocity = c2 * r2 * (global_best_position - particle_positions)
    particle_velocities = inertia_weight * particle_velocities + cognitive_velocity + social_velocity
    particle_positions += particle_velocities

    if (iter + 1) % 10 == 0:
        print(f"Iteration {iter + 1}/{n_iters}, Global Best Score: {global_best_score:.6f}")

# Denormalize learned parameters
A_best_norm, B_best_norm = reshape_params(global_best_position)
A_best, B_best = denormalize_params(A_best_norm, B_best_norm)

print("\nLearned Parameters (Denormalized):")
print("A:\n", A_best)
print("B:\n", B_best)
