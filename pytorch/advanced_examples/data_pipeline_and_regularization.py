"""
DATA PIPELINE + BATCHING + BATCHNORM + DROPOUT + TRAIN/EVAL
===========================================================

REVISION QUESTIONS & ANSWERS
----------------------------

Q1: What is a Dataset in PyTorch?
A:
A Dataset is a Python class that defines how to access individual samples.
It must implement:
    __len__()
    __getitem__(index)

Q2: What is a DataLoader?
A:
DataLoader wraps a Dataset and:
    - Splits it into batches
    - Shuffles data
    - Handles parallel loading
    - Iterates batch-by-batch

Q3: What is BatchNorm?
A:
BatchNorm normalizes activations within a batch:
    (x - mean) / sqrt(var + epsilon)
Then applies learnable scale and shift.
It stabilizes training and reduces internal covariate shift.

Q4: What is Dropout?
A:
Dropout randomly sets some activations to zero during training.
It prevents overfitting by forcing redundancy in representation.

Q5: Why train() and eval()?
A:
train() enables:
    - Dropout active
    - BatchNorm uses batch statistics

eval() enables:
    - Dropout disabled
    - BatchNorm uses running averages

This is CRITICAL during inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# 1. CUSTOM DATASET
# ---------------------------------------------------------

class CustomDataset(Dataset):
    """
    Simulated dataset with:
        10,000 samples
        25 features
        1 target value
    """

    def __init__(self):
        self.X = torch.randn(10000, 25)
        self.y = torch.randn(10000, 1)

    def __len__(self):
        # Returns number of samples
        return len(self.X)

    def __getitem__(self, index):
        # Returns single sample
        return self.X[index], self.y[index]


dataset = CustomDataset()

# ---------------------------------------------------------
# 2. DATALOADER (BATCHING HAPPENS HERE)
# ---------------------------------------------------------

dataloader = DataLoader(
    dataset,
    batch_size=100,
    shuffle=True
)

# Now:
# 10,000 samples / batch_size=100 → 100 batches per epoch


# ---------------------------------------------------------
# 3. MODEL WITH BATCHNORM + DROPOUT
# ---------------------------------------------------------

class AdvancedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(25, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)

        # BatchNorm normalizes activations across batch
        x = self.bn1(x)

        x = self.relu(x)

        # Dropout randomly zeros neurons (only in train mode)
        x = self.dropout(x)

        x = self.layer2(x)
        return x


model = AdvancedModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ---------------------------------------------------------
# 4. TRAINING LOOP (FULL FLOW)
# ---------------------------------------------------------

epochs = 2

for epoch in range(epochs):

    model.train()  # IMPORTANT

    total_loss = 0

    for batch_X, batch_y in dataloader:

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(predictions, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# ---------------------------------------------------------
# 5. EVALUATION MODE
# ---------------------------------------------------------

model.eval()

# Disable gradient tracking during evaluation
with torch.no_grad():

    sample_X, _ = dataset[0]
    sample_X = sample_X.unsqueeze(0)

    prediction = model(sample_X)

    print("\nPrediction in eval mode:", prediction.item())


# ---------------------------------------------------------
# WHAT HAPPENS INTERNALLY?
# ---------------------------------------------------------

"""
BatchNorm during train():
    - Computes mean & variance from current batch
    - Updates running statistics

BatchNorm during eval():
    - Uses running mean & variance (learned during training)

Dropout during train():
    - Randomly zeros activations

Dropout during eval():
    - Disabled (no neurons dropped)

DataLoader:
    - Fetches 100 samples
    - Feeds into model
    - 100 updates per epoch

Complete Flow per Batch:
    Forward
    Loss
    Backward
    Optimizer step

Q6: Why is BatchNorm usually applied BEFORE ReLU?

A:
BatchNorm normalizes the linear layer outputs to have stable mean and variance.
If we apply ReLU first, negative values become zero and the distribution
becomes skewed and non-symmetric.

By applying BatchNorm before ReLU:
    - We normalize the full activation distribution.
    - We maintain stable variance before non-linearity.
    - Training becomes more stable and gradients flow better.

Typical pattern:
    Linear → BatchNorm → ReLU

This ensures normalization happens on the full signal.
Q7: Why must we call model.eval() during inference?

A:
Some layers behave differently during training and inference.

During training:
    - Dropout randomly disables neurons.
    - BatchNorm uses current batch statistics.

During evaluation:
    - Dropout must be disabled.
    - BatchNorm must use learned running mean and variance.

If we don’t call model.eval():
    - Dropout will randomly zero neurons during inference.
    - Predictions will change every forward pass.
    - BatchNorm will use batch statistics instead of learned population stats.

This makes predictions unstable and incorrect.
Q8: What happens if we forget model.eval() and use Dropout during testing?

A:
Dropout randomly removes neurons.

During testing, we want deterministic predictions.
If Dropout is active:
    - Every prediction becomes random.
    - Output changes every run.
    - Model underestimates its learned capacity.

Training uses Dropout to prevent overfitting.
Inference must use full network capacity.

Therefore:
    Always call model.eval() before inference.
"""