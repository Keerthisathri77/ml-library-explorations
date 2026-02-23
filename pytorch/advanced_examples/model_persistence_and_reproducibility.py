"""
MODEL PERSISTENCE + REPRODUCIBILITY
===================================

REVISION QUESTIONS & ANSWERS
----------------------------

Q1: What is state_dict()?
A:
state_dict() is a dictionary containing all learnable parameters
(weights and biases) of a model.

Q2: Why do we save state_dict instead of entire model?
A:
Saving state_dict is safer and more flexible.
It avoids dependency on exact Python class structure.

Q3: What should we save during training?
A:
- model state_dict
- optimizer state_dict
- current epoch
- loss value

Q4: Why is reproducibility important?
A:
Deep learning uses randomness (weight init, shuffling, dropout).
Setting seeds ensures experiments can be repeated.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


# ---------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # If using CUDA (not in your case)
    # torch.cuda.manual_seed_all(seed)

set_seed(42)


# ---------------------------------------------------------
# 2. SIMPLE MODEL
# ---------------------------------------------------------

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(25, 1)

    def forward(self, x):
        return self.fc(x)


model = SmallModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# ---------------------------------------------------------
# 3. SIMULATED TRAINING STEP
# ---------------------------------------------------------

X = torch.randn(100, 25)
y = torch.randn(100, 1)

optimizer.zero_grad()
output = model(X)
loss = criterion(output, y)
loss.backward()
optimizer.step()


# ---------------------------------------------------------
# 4. SAVE CHECKPOINT
# ---------------------------------------------------------

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item()
}

torch.save(checkpoint, "checkpoint.pth")

print("Model saved.")


# ---------------------------------------------------------
# 5. LOAD CHECKPOINT
# ---------------------------------------------------------

new_model = SmallModel()
new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)

loaded_checkpoint = torch.load("checkpoint.pth")

new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

print("Model loaded successfully.")


# ---------------------------------------------------------
# WHY THIS MATTERS
# ---------------------------------------------------------

"""
In real training:

- Training may take hours or days.
- You must resume from checkpoint if interrupted.
- You may want to deploy trained weights later.
- You may want reproducible experiments.

state_dict is the backbone of model persistence in PyTorch.
"""