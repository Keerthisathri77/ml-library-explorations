"""
OPTIMIZATION AND TRAINING STABILITY
===================================

REVISION QUESTIONS & ANSWERS
----------------------------

Q1: What is SGD?
A:
Stochastic Gradient Descent updates parameters using:
    W = W - lr * gradient

Q2: What is Momentum?
A:
Momentum accumulates past gradients to smooth updates.
It helps move faster in consistent directions.

Q3: What is Adam?
A:
Adam combines:
    - Momentum (first moment)
    - Adaptive learning rates (second moment)

Q4: What is gradient clipping?
A:
It prevents exploding gradients by limiting gradient magnitude.

Q5: What is a learning rate scheduler?
A:
It changes learning rate during training to improve convergence.
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------
# 1. SIMPLE MODEL
# ---------------------------------------------------------

class SmallDeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


model = SmallDeepNet()
criterion = nn.MSELoss()


# ---------------------------------------------------------
# 2. SGD vs ADAM
# ---------------------------------------------------------

sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

adam_optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
SGD update rule:

    v = momentum * v - lr * grad
    W = W + v

Adam update rule:

    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad^2

    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)

    W = W - lr * m_hat / (sqrt(v_hat) + epsilon)

Adam adapts learning rate per parameter.
"""


# ---------------------------------------------------------
# 3. GRADIENT CLIPPING
# ---------------------------------------------------------

X = torch.randn(100, 25)
y = torch.randn(100, 1)

optimizer = adam_optimizer

optimizer.zero_grad()

output = model(X)
loss = criterion(output, y)
loss.backward()

# Compute gradient norm BEFORE clipping
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

total_norm = total_norm ** 0.5
print("Gradient norm before clipping:", total_norm)

# Clip gradients to max norm of 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Compute gradient norm AFTER clipping
total_norm_after = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm_after += param_norm.item() ** 2

total_norm_after = total_norm_after ** 0.5
print("Gradient norm after clipping:", total_norm_after)

optimizer.step()


# ---------------------------------------------------------
# 4. LEARNING RATE SCHEDULER
# ---------------------------------------------------------

optimizer = optim.Adam(model.parameters(), lr=0.01)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

"""
StepLR:
    Every 5 epochs,
    learning rate = learning_rate * 0.5
"""

for epoch in range(10):

    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    scheduler.step()

    print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]}")


# ---------------------------------------------------------
# WHY THESE MATTER
# ---------------------------------------------------------

"""
SGD:
    Simple, stable, good generalization.

Adam:
    Faster convergence.
    Good for noisy gradients.

Gradient Clipping:
    Prevents exploding gradients in deep or RNN models.

Learning Rate Scheduling:
    Large LR initially (fast learning).
    Smaller LR later (fine tuning).
"""