import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SimplePendulumNN(nn.Module):
    def __init__(self):
        super(SimplePendulumNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )

    def forward(self, t):
        return self.layer(t)

    def loss_fn(self, t, theta_obs):
        theta_pred = self.forward(t)
        mse_loss = nn.MSELoss()(theta_pred, theta_obs)

        theta_pred_leaf = theta_pred.detach().clone().requires_grad_(True)
        
        theta_dot_dot = torch.autograd.grad(
            outputs=theta_pred_leaf,
            inputs=t,
            grad_outputs=torch.ones_like(theta_pred_leaf),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Handle None
        if theta_dot_dot is None:
            theta_dot_dot = torch.zeros_like(theta_pred_leaf)
        
        g = 9.81  # gravity
        l = 1.0  # length

        physics_loss = torch.mean((theta_dot_dot + (g/l) * torch.sin(theta_pred_leaf)) ** 2)
        loss = mse_loss + physics_loss
        return loss, mse_loss, physics_loss

# Generate some example data
t_data = np.linspace(0, 10, 100).reshape(-1, 1)
theta_data = np.sin(t_data)

t_tensor = torch.tensor(t_data, dtype=torch.float32, requires_grad=True)
theta_tensor = torch.tensor(theta_data, dtype=torch.float32)

# Initialize model and optimizer
model = SimplePendulumNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(5000):
    optimizer.zero_grad()
    loss, mse_loss, physics_loss = model.loss_fn(t_tensor, theta_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, MSE Loss: {mse_loss.item()}, Physics Loss: {physics_loss.item()}')

# Plot results
t_test_data = np.linspace(0, 20, 100).reshape(-1, 1)
t_test_tensor = torch.tensor(t_test_data, dtype=torch.float32, requires_grad=True)
theta_pred = model(t_test_tensor).detach().numpy()

plt.figure()
plt.plot(t_test_data, theta_pred, label='Predicted')
plt.plot(t_data, theta_data, label='True')
plt.legend()
plt.show()
