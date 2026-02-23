"""
TENSOR FUNDAMENTALS IN PYTORCH
================================

This script demonstrates:

1. Tensor creation
2. Shape inspection
3. Matrix multiplication
4. Broadcasting rules
5. requires_grad mechanics
6. Basic gradient computation

Everything is explained in comments in detail.
"""

import torch

# -------------------------------
# 1. WHAT IS A TENSOR?
# -------------------------------
# A tensor is a multi-dimensional array.
# It is the fundamental data structure in PyTorch.
#
# Scalars  -> 0D tensor
# Vector   -> 1D tensor
# Matrix   -> 2D tensor
# Higher   -> ND tensor

# Create a scalar (0D tensor)
scalar = torch.tensor(5.0)
print("Scalar:", scalar)
print("Scalar shape:", scalar.shape)  # shape is empty because it's 0D

# Create a vector (1D tensor)
vector = torch.tensor([1.0, 2.0, 3.0])
print("\nVector:", vector)
print("Vector shape:", vector.shape)  # (3,)

# Create a matrix (2D tensor)
matrix = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]])
print("\nMatrix:\n", matrix)
print("Matrix shape:", matrix.shape)  # (2, 2)


# -------------------------------
# 2. RANDOM TENSOR CREATION
# -------------------------------
# Common in ML: initialize weights randomly

random_tensor = torch.randn(3, 4)  # 3 rows, 4 columns
print("\nRandom tensor:\n", random_tensor)
print("Random tensor shape:", random_tensor.shape)


# -------------------------------
# 3. MATRIX MULTIPLICATION
# -------------------------------
# Linear algebra foundation of neural networks.
#
# If:
# A is (m x n)
# B is (n x p)
# Then:
# A @ B is (m x p)

A = torch.randn(2, 3)
B = torch.randn(3, 4)

C = A @ B  # matrix multiplication
print("\nMatrix multiplication result shape:", C.shape)  # (2,4)

# Internally:
# Each row of A is dot-producted with each column of B.


# -------------------------------
# 4. BROADCASTING
# -------------------------------
# Broadcasting allows operations between tensors of different shapes.
#
# Example:
# (3,4) + (4,) is valid.
# PyTorch expands smaller tensor automatically.

X = torch.ones(3, 4)
bias = torch.tensor([1.0, 2.0, 3.0, 4.0])  # shape (4,)

Y = X + bias
print("\nBroadcast result shape:", Y.shape)

# Explanation:
# bias is treated as if repeated across rows:
# [[1,2,3,4],
#  [1,2,3,4],
#  [1,2,3,4]]


# -------------------------------
# 5. requires_grad
# -------------------------------
# When building neural networks,
# we want gradients with respect to parameters.
#
# requires_grad=True tells PyTorch:
# "Track operations on this tensor for backpropagation."

x = torch.tensor(3.0, requires_grad=True)

# Define function:
# y = x^2
y = x ** 2

# Compute gradient
y.backward()

print("\nValue of x:", x.item())
print("Computed gradient dy/dx:", x.grad.item())

# Why is gradient = 6?
# Because derivative of x^2 is 2x
# At x=3:
# 2 * 3 = 6


# -------------------------------
# 6. MULTI-VARIABLE EXAMPLE
# -------------------------------

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)

# Define function:
# f = a*b + b^2
f = a * b + b ** 2

# Backpropagation
f.backward()

print("\nGradient wrt a:", a.grad.item())
print("Gradient wrt b:", b.grad.item())

# Let’s verify manually:
# f = ab + b^2
# df/da = b
# df/db = a + 2b
#
# At a=2, b=4:
# df/da = 4
# df/db = 2 + 8 = 10
