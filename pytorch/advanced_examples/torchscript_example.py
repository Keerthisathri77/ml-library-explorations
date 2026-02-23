import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        if x.sum() > 0:
            return self.linear(x)
        else:
            return -self.linear(x)

model = MyModel()

scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pt")

print("Model scripted and saved.")
