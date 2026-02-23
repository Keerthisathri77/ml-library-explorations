"""
MULTI-LAYER NETWORKS AND NON-LINEARITY
======================================

This script explains:

1. Why linear-only models are limited
2. How to stack layers
3. How activation functions introduce non-linearity
4. How gradients flow through activation functions
5. What happens during forward and backward pass

We simulate everything with random data.
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------
# 1. LINEAR MODEL (NO NON-LINEARITY)
# ---------------------------------------------------------

class LinearOnlyModel(nn.Module):
    """
    This model is:

        y = W2(W1x)

    Even though it has 2 layers,
    because no activation function exists,
    the entire system is still linear.

    Why?

    Because:
        W2(W1x) = (W2W1)x

    That is still one linear transformation.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(25, 16)
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# ---------------------------------------------------------
# 2. NON-LINEAR MODEL
# ---------------------------------------------------------

class NonLinearModel(nn.Module):
    """
    This model introduces ReLU activation.

    Now:

        y = W2(ReLU(W1x))

    ReLU is non-linear.

    So:
        You CANNOT collapse this into a single matrix multiplication.

    This increases expressive power.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(25, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)   # non-linearity happens here
        x = self.layer2(x)
        return x


# ---------------------------------------------------------
# 3. WHY ReLU?
# ---------------------------------------------------------

"""
ReLU(x) = max(0, x)

Derivative:

If x > 0:
    derivative = 1
If x <= 0:
    derivative = 0

This allows gradient to flow when active,
but blocks gradient when neuron is inactive.

Compared to sigmoid:
- ReLU reduces vanishing gradient problem
"""


# ---------------------------------------------------------
# 4. SIMULATED TRAINING STEP
# ---------------------------------------------------------

model = NonLinearModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

X = torch.randn(100, 25)
y_true = torch.randn(100, 1)

optimizer.zero_grad()

predictions = model(X)
loss = criterion(predictions, y_true)

print("Loss:", loss.item())

loss.backward()

print("\nGradient of first layer weight:")
print(model.layer1.weight.grad)

optimizer.step()


# ---------------------------------------------------------
# 5. WHAT HAPPENS DURING BACKWARD?
# ---------------------------------------------------------

"""
Backward flow:

Loss
 ↓
Layer2 gradient
 ↓
ReLU gradient (masking negatives)
 ↓
Layer1 gradient

If ReLU input was negative,
its gradient is 0,
so gradient does not flow backward through that neuron.

This is called "dead neuron" effect if it stays negative.
"""