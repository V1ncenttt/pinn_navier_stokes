import torch
import torch.nn as nn
import numpy as np
import scipy.io
from fenics import Mesh

# Function to read data from .mat files
def read_mat(filename):
    data = scipy.io.loadmat(filename)
    return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NavierStokesNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_layer1 = nn.Linear(3, 20)  # 3 input features: x, y, t
        self.hidden_layer2 = nn.Linear(20, 20)
        self.hidden_layer3 = nn.Linear(20, 20)
        self.hidden_layer4 = nn.Linear(20, 20)
        self.hidden_layer5 = nn.Linear(20, 20)
        self.hidden_layer6 = nn.Linear(20, 20)
        self.hidden_layer7 = nn.Linear(20, 20)
        self.hidden_layer8 = nn.Linear(20, 20)
        self.hidden_layer9 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 2)  # 2 output features: u, v (velocity, pressure)

    def forward(self, x, y, t):
        out = torch.cat([x, y, t], dim=-1)  # Concatenate along the last dimension
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        out = torch.tanh(self.hidden_layer3(out))
        out = torch.tanh(self.hidden_layer4(out))
        out = torch.tanh(self.hidden_layer5(out))
        out = torch.tanh(self.hidden_layer6(out))
        out = torch.tanh(self.hidden_layer7(out))
        out = torch.tanh(self.hidden_layer8(out))
        out = torch.tanh(self.hidden_layer9(out))
        out = self.output_layer(out)
        return out

def function(model, x, y, t):
    model_output = model(x, y, t)
    psi, p = model_output[:, 0:1], model_output[:, 1:2]
    nu = 0.01

    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -1 * torch.autograd.grad(psi.sum(), x, create_graph=True)[0]

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]

    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]

    f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return u, v, p, f, g

# Load data and normalize it
# Reading velocity data
velocity_data = read_mat('navier_stokes_cylinder/velocity.mat')['velocity']
# Reading pressure data
pressure_data = read_mat('navier_stokes_cylinder/pressure.mat')['pressure']

# Load the mesh from the .xml file using FEniCS
mesh = Mesh('navier_stokes_cylinder/cylinder.xml')
coordinates = mesh.coordinates()

# Verify the shapes of the loaded data
print(f"Velocity data shape: {velocity_data.shape}")
print(f"Pressure data shape: {pressure_data.shape}")
print(f"Coordinates shape: {coordinates.shape}")

# Define the domain boundaries and time span
x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
t_min, t_max = 0, 1  # Assuming the time range

# Extract initial condition data
x_init = coordinates[:, 0]
y_init = coordinates[:, 1]
t_init = np.zeros_like(x_init, dtype=np.float32)
# Use the first time step for initial conditions
u_init = velocity_data[0, :coordinates.shape[0]]  # Match the coordinate shape
v_init = velocity_data[1, :coordinates.shape[0]]  # Match the coordinate shape
p_init = pressure_data[0, :coordinates.shape[0]]  # Match the coordinate shape

# Verify the shapes of initial condition data
print(f"Initial conditions u_init shape: {u_init.shape}")
print(f"Initial conditions v_init shape: {v_init.shape}")
print(f"Initial conditions p_init shape: {p_init.shape}")

# Extract boundary condition data (example for one time step)
t_some_time = 5  # Replace with an appropriate time step index
x_bound = coordinates[:, 0]
y_bound = coordinates[:, 1]
t_bound = np.ones_like(x_bound, dtype=np.float32) * t_some_time
u_bound = velocity_data[t_some_time, :coordinates.shape[0]]  # Match the coordinate shape
v_bound = velocity_data[t_some_time, :coordinates.shape[0]]  # Match the coordinate shape
p_bound = pressure_data[t_some_time, :coordinates.shape[0]]  # Match the coordinate shape

# Generate collocation points
num_collocation_points = 10000
x_collocation = np.random.uniform(x_min, x_max, num_collocation_points)
y_collocation = np.random.uniform(y_min, y_max, num_collocation_points)
t_collocation = np.random.uniform(t_min, t_max, num_collocation_points)

# Convert all data to torch tensors and move to the appropriate device
x_init = torch.tensor(x_init, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
y_init = torch.tensor(y_init, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
t_init = torch.tensor(t_init, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
u_init = torch.tensor(u_init, dtype=torch.float32).unsqueeze(1).to(device)
v_init = torch.tensor(v_init, dtype=torch.float32).unsqueeze(1).to(device)
p_init = torch.tensor(p_init, dtype=torch.float32).unsqueeze(1).to(device)

x_bound = torch.tensor(x_bound, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
y_bound = torch.tensor(y_bound, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
t_bound = torch.tensor(t_bound, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
u_bound = torch.tensor(u_bound, dtype=torch.float32).unsqueeze(1).to(device)
v_bound = torch.tensor(v_bound, dtype=torch.float32).unsqueeze(1).to(device)
p_bound = torch.tensor(p_bound, dtype=torch.float32).unsqueeze(1).to(device)

x_collocation = torch.tensor(x_collocation, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
y_collocation = torch.tensor(y_collocation, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
t_collocation = torch.tensor(t_collocation, dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)

# Initialize the network
nsnet = NavierStokesNet().to(device)
cost_func = nn.MSELoss()

# Use AdamW optimizer for initial epochs
optimizer = torch.optim.AdamW(nsnet.parameters(), lr=0.001)

# Training loop with AdamW
num_initial_epochs = 50000
for epoch in range(num_initial_epochs):
    optimizer.zero_grad()
    
    # Initial conditions
    u_pred_init, v_pred_init, p_pred_init, _, _ = function(nsnet, x_init, y_init, t_init)
    loss_ic = cost_func(u_pred_init, u_init) + cost_func(v_pred_init, v_init) + cost_func(p_pred_init, p_init)
    
    # Boundary conditions
    u_pred_bound, v_pred_bound, p_pred_bound, _, _ = function(nsnet, x_bound, y_bound, t_bound)
    loss_bc = cost_func(u_pred_bound, u_bound) + cost_func(v_pred_bound, v_bound) + cost_func(p_pred_bound, p_bound)
    
    # Collocation points
    _, _, _, f_pred, g_pred = function(nsnet, x_collocation, y_collocation, t_collocation)
    loss_pde = cost_func(f_pred, torch.zeros_like(f_pred)) + cost_func(g_pred, torch.zeros_like(g_pred))
    
    # Total loss
    loss = loss_ic + loss_bc + loss_pde
    
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'AdamW - Epoch {epoch}, Loss: {loss.item()}')

# Switch to LBFGS optimizer for fine-tuning
optimizer = torch.optim.LBFGS(nsnet.parameters(), lr=0.1, max_iter=10000, tolerance_grad=1e-9, tolerance_change=1e-9)

# Define the closure function for LBFGS
def closure():
    optimizer.zero_grad()
    
    # Initial conditions
    u_pred_init, v_pred_init, p_pred_init, _, _ = function(nsnet, x_init, y_init, t_init)
    loss_ic = cost_func(u_pred_init, u_init) + cost_func(v_pred_init, v_init) + cost_func(p_pred_init, p_init)
    
    # Boundary conditions
    u_pred_bound, v_pred_bound, p_pred_bound, _, _ = function(nsnet, x_bound, y_bound, t_bound)
    loss_bc = cost_func(u_pred_bound, u_bound) + cost_func(v_pred_bound, v_bound) + cost_func(p_pred_bound, p_bound)
    
    # Collocation points
    _, _, _, f_pred, g_pred = function(nsnet, x_collocation, y_collocation, t_collocation)
    loss_pde = cost_func(f_pred, torch.zeros_like(f_pred)) + cost_func(g_pred, torch.zeros_like(g_pred))
    
    # Total loss
    loss = loss_ic + loss_bc + loss_pde
    
    loss.backward()
    return loss

# Training loop with LBFGS
num_epochs_lbfgs = 500
for epoch in range(num_epochs_lbfgs):
    loss = optimizer.step(closure)
    
    if epoch % 100 == 0:
        print(f'LBFGS - Epoch {epoch}, Loss: {loss.item()}')
