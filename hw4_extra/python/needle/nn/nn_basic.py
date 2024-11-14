"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad = True, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad = True, device=device, dtype=dtype).transpose())
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if len(X.shape) == 2:
            # X has shape (batch_size, input_dim)
            output = ops.matmul(X, self.weight)
            if self.bias is not None:
                output += self.bias.broadcast_to(output.shape)
            return output
        elif len(X.shape) == 3:
            # X has shape (batch_size, seq_len, input_dim)
            batch_size, seq_len, input_dim = X.shape
            total_samples = batch_size * seq_len
            X_reshaped = X.reshape((total_samples, input_dim))
            output = ops.matmul(X_reshaped, self.weight)
            if self.bias is not None:
                output += self.bias.broadcast_to(output.shape)
            output = output.reshape((batch_size, seq_len, self.out_features))
            return output
        ### END YOUR SOLUTION

from functools import reduce

class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        size = reduce(lambda a, b: a * b, X.shape)
        return X.reshape((X.shape[0], size // X.shape[0]))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot_y = init.one_hot(logits.shape[1], y)
        lse = ops.logsumexp(logits, axes=(1,)) / logits.shape[0]
        z_y = ops.summation(one_hot_y * logits / logits.shape[0])
        return ops.summation(lse) - z_y        
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device))
        self.running_mean = init.zeros(dim, device=device)
        self.running_var = init.ones(dim, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_mean = (x.sum((0,)) / x.shape[0]) # (num_features, )
            batch_mean_vec = batch_mean.reshape((1, x.shape[1])) # (1, num_features)
            batch_var = (((x - batch_mean_vec.broadcast_to(x.shape))**2).sum((0,)) / x.shape[0]) # (num_features, )
            batch_var_vec = batch_var.reshape((1, x.shape[1])) # (1, num_features)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            norm = (x - batch_mean_vec.broadcast_to(x.shape)) / (batch_var_vec.broadcast_to(x.shape) + self.eps)**0.5
            return self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape) * norm \
                 + self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)
        else:
            norm = (x - self.running_mean.reshape((1, x.shape[1])).broadcast_to(x.shape)) / (self.running_var.reshape((1, x.shape[1])).broadcast_to(x.shape) + self.eps)**0.5
            return self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape) * norm + self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # mean = (ops.summation(x, (1,))/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # var = (ops.summation(((x - mean)**2), (1,))/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # x_hat = (x - mean) / (var + self.eps) ** 0.5
        # return self.weight.broadcast_to(x.shape) * x_hat + self.bias.broadcast_to(x.shape)

        # Compute mean and variance over the last dimension
        axes = (len(x.shape) - 1,)  # Normalize over the last dimension
        mean = ops.summation(x, axes=axes) / x.shape[-1]
        mean = mean.reshape(mean.shape + (1,)).broadcast_to(x.shape)

        var = ops.summation((x - mean) ** 2, axes=axes) / x.shape[-1]
        var = var.reshape(var.shape + (1,)).broadcast_to(x.shape)

        x_hat = (x - mean) / (var + self.eps) ** 0.5

        broadcast_shape = (1,) * (len(x.shape) - 1) + (self.dim,)
        weight = self.weight.reshape(broadcast_shape).broadcast_to(x.shape)
        bias = self.bias.reshape(broadcast_shape).broadcast_to(x.shape)

        return weight * x_hat + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5, device=None, dtype="float32"):
        super().__init__()
        self.p = p
        self.device = device
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=self.device, dtype=self.dtype)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
