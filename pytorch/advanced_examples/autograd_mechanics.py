"""
AUTOGRAD MECHANICS IN PYTORCH
=============================

This script explains:

1. Leaf vs Non-Leaf tensors
2. grad_fn inspection
3. Gradient accumulation behavior
4. Calling backward twice
5. retain_graph
6. detach()
7. torch.no_grad()

Read comments carefully.
Run the script.
Observe outputs.
"""

import torch


# ---------------------------------------------------------
# 1. LEAF VS NON-LEAF TENSORS
# ---------------------------------------------------------

# Leaf tensor: created directly by user with requires_grad=True
a = torch.tensor(2.0, requires_grad=True)

# Non-leaf tensor: result of an operation
b = a * 3
c = b * 4

print("Is 'a' leaf?", a.is_leaf)  # True
print("Is 'b' leaf?", b.is_leaf)  # False
print("Is 'c' leaf?", c.is_leaf)  # False

# Only leaf tensors store gradients by default


# ---------------------------------------------------------
# 2. INSPECTING COMPUTATION GRAPH
# ---------------------------------------------------------

print("\nGradient function of b:", b.grad_fn)
print("Gradient function of c:", c.grad_fn)

# grad_fn tells you what backward operation will be used


# ---------------------------------------------------------
# 3. BACKWARD PASS
# ---------------------------------------------------------

c.backward()

print("\nGradient of a after backward:", a.grad)

# Why 12?
# c = (a*3)*4 = 12a
# dc/da = 12


# ---------------------------------------------------------
# 4. GRADIENT ACCUMULATION
# ---------------------------------------------------------

# Call backward again WITHOUT zeroing gradients

c = (a * 3) * 4
c.backward()

print("Gradient of a after second backward:", a.grad)

# Notice:
# It is now 24.
# Because gradients accumulate.
# PyTorch does: a.grad += new_gradient


# ---------------------------------------------------------
# 5. RESETTING GRADIENTS
# ---------------------------------------------------------

a.grad.zero_()

print("Gradient after manual reset:", a.grad)


# ---------------------------------------------------------
# 6. RETAINING NON-LEAF GRADIENTS
# ---------------------------------------------------------

a = torch.tensor(2.0, requires_grad=True)
b = a * 3

b.retain_grad()  # Tell PyTorch to keep gradient for non-leaf

c = b * 4
c.backward()

print("\nGradient of a:", a.grad)
print("Gradient of b:", b.grad)

# Now b.grad is populated because we explicitly retained it.


# ---------------------------------------------------------
# 7. DETACHING FROM GRAPH
# ---------------------------------------------------------

a = torch.tensor(2.0, requires_grad=True)
b = a * 3

detached = b.detach()

print("\nDetached requires_grad?", detached.requires_grad)

# detached tensor is removed from computation graph
# No gradient tracking


# ---------------------------------------------------------
# 8. torch.no_grad()
# ---------------------------------------------------------

a = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    b = a * 5

print("\nDoes b track gradients?", b.requires_grad)

# torch.no_grad() is used during inference
# It disables gradient tracking to save memory and compute