"""
INITIALIZATION AND GRADIENT FLOW
================================

REVISION QUESTIONS & ANSWERS
-----------------------------

Q1: Why do deep networks suffer from vanishing or exploding gradients?

A:
Because gradients are computed using chain rule.
Each layer multiplies gradients by its weight matrix.
Repeated multiplication causes values to either shrink (<1)
or grow (>1) exponentially with depth.


Q2: What is the goal of weight initialization?

A:
To preserve variance of activations and gradients across layers,
so that signals neither explode nor vanish.


Q3: Why is Xavier initialization good for sigmoid/tanh?

A:
Because those activations are symmetric and squash outputs.
Xavier keeps variance balanced between input and output dimensions:
Var(W) = 2 / (n_in + n_out)


Q4: Why is Kaiming initialization good for ReLU?

A:
ReLU zeroes out negative values (~half activations).
To compensate, Kaiming increases variance:
Var(W) = 2 / n_in
This preserves forward signal strength.


Q5: Why does ReLU reduce vanishing gradient problem?

A:
Because derivative is 1 for positive inputs.
Unlike sigmoid, which has derivative < 1 everywhere,
ReLU allows stronger gradient flow.
"""

import torch
import torch.nn as nn


class DeepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# Create model
model = DeepNetwork()

X = torch.randn(100, 25)
y = torch.randn(100, 1)

criterion = nn.MSELoss()

output = model(X)
loss = criterion(output, y)
loss.backward()

print("Default init gradient magnitude:",
      model.net[0].weight.grad.abs().mean().item())


# Apply Kaiming initialization
model2 = DeepNetwork()

for layer in model2.modules():
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)

output2 = model2(X)
loss2 = criterion(output2, y)
loss2.backward()

print("Kaiming init gradient magnitude:",
      model2.net[0].weight.grad.abs().mean().item())