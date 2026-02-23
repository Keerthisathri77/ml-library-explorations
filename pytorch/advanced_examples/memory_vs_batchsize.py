

"""
EXPERIMENT 1: MEMORY VS BATCH SIZE
==================================

This experiment measures:

1. How memory usage increases with batch size
2. How runtime changes with batch size
3. Why large batch sizes consume more memory

Key Insight:
Memory scales roughly linearly with batch size
because intermediate activations scale with input size.
"""

import torch
import torch.nn as nn
import time
import tracemalloc


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


model = SimpleModel()
criterion = nn.MSELoss()

batch_sizes = [32, 64, 128, 256, 512]

for batch_size in batch_sizes:

    # Track memory
    tracemalloc.start()

    X = torch.randn(batch_size, 25)
    y = torch.randn(batch_size, 1)

    start_time = time.time()

    output = model(X)
    loss = criterion(output, y)
    loss.backward()

    end_time = time.time()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nBatch Size: {batch_size}")
    print(f"Peak Memory (KB): {peak / 1024:.2f}")
    print(f"Runtime (ms): {(end_time - start_time)*1000:.2f}")

"""
Observations:

- Larger batch sizes increase memory.
- Runtime increases but may scale non-linearly.
- On GPU, memory pressure is more pronounced.
"""