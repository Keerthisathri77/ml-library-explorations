"""
EXPERIMENT 5: TORCHSCRIPT + COMPILATION

Demonstrates:

1. Model scripting
2. Saving compiled model
3. Inference comparison
"""

import torch
import torch.nn as nn
import time


class ScriptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


model = ScriptModel()
X = torch.randn(1000, 25)

# Normal inference
start = time.time()
_ = model(X)
end = time.time()
print("Normal inference time:", end - start)

# Scripted model
scripted_model = torch.jit.script(model)

start = time.time()
_ = scripted_model(X)
end = time.time()
print("Scripted inference time:", end - start)

scripted_model.save("scripted_model.pt")

"""
Scripted model:
- Removes Python overhead
- Allows C++ deployment
- Enables graph-level optimizations
"""