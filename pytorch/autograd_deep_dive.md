# Autograd Deep Dive

PyTorch's autograd engine implements reverse-mode automatic differentiation. It enables efficient gradient computation for high-dimensional parameter spaces.

## 1. Computation Graph Construction

When tensors with `requires_grad=True` are used in operations, PyTorch records those operations as nodes in a computation graph.

Each tensor contains:
- `.grad_fn` reference
- Parent function references

The graph is constructed dynamically during forward execution.

## 2. Reverse-Mode Differentiation

Reverse-mode differentiation computes gradients of scalar outputs with respect to many inputs efficiently.

Given:
L = f(x)

Autograd computes:
dL/dx

by traversing graph in reverse topological order.

## 3. Leaf vs Non-Leaf Tensors

Leaf tensors:
- Created directly by user.
- Store gradients in `.grad`.

Non-leaf tensors:
- Results of operations.
- Do not store gradients unless `.retain_grad()` is called.

## 4. Gradient Accumulation

Gradients accumulate by default:

param.grad += computed_gradient

This allows gradient accumulation across mini-batches.

## 5. Graph Retention

After backward:
Graph is freed unless `retain_graph=True`.

This prevents memory leaks.

## 6. Custom Autograd Functions

Users can define:

class CustomFn(torch.autograd.Function)

This allows manual forward and backward definition.

## 7. Limitations

- Only differentiates through PyTorch ops.
- No automatic symbolic simplification.
- Memory intensive for deep graphs.

## Summary

Autograd is:
- Dynamic
- Exact (not numerical)
- Reverse-mode
- Efficient for deep learning