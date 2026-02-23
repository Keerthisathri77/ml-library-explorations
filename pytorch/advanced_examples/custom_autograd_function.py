import torch

class CustomSquare(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * 2 * input
        return grad_input


x = torch.tensor([3.0], requires_grad=True)
y = CustomSquare.apply(x)
y.backward()

print(x.grad)
