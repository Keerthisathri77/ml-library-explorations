import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(1000, 2048),
    nn.ReLU(),
    nn.Linear(2048, 10)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

scaler = torch.cuda.amp.GradScaler()

for step in range(5):
    inputs = torch.randn(64, 1000).to(device)
    targets = torch.randn(64, 10).to(device)

    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Step {step}, Loss: {loss.item()}")
