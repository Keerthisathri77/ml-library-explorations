"""
GRADIENT DEBUGGING + HOOKS
==========================

REVISION QUESTIONS & ANSWERS
----------------------------

Q1: What is a hook?
A:
A hook is a function attached to a tensor or module
that executes during forward or backward pass.

Q2: Why use gradient hooks?
A:
To inspect gradient values.
To debug vanishing/exploding gradients.
To detect NaNs.

Q3: When should you use this?
A:
When training becomes unstable.
When loss becomes NaN.
When gradients are zero unexpectedly.
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------
# 1. MODEL
# ---------------------------------------------------------

class DebugNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = DebugNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# ---------------------------------------------------------
# 2. REGISTER GRADIENT HOOK
# ---------------------------------------------------------

def gradient_hook(grad):
    print("Gradient mean:", grad.mean().item())
    print("Gradient max:", grad.max().item())
    print("Gradient min:", grad.min().item())
    return grad  # must return gradient


# Attach hook to first layer weights
model.fc1.weight.register_hook(gradient_hook)


# ---------------------------------------------------------
# 3. TRAINING STEP
# ---------------------------------------------------------

X = torch.randn(100, 25)
y = torch.randn(100, 1)

optimizer.zero_grad()
output = model(X)
loss = criterion(output, y)

print("Loss:", loss.item())

loss.backward()  # hook executes here

optimizer.step()


# ---------------------------------------------------------
# 4. DETECT NAN GRADIENTS
# ---------------------------------------------------------

for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN detected in {name}")