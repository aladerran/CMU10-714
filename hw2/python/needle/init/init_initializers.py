import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    return rand(fan_in, fan_out, low=gain * (-math.sqrt(6 / (fan_in + fan_out))), high=gain * (math.sqrt(6 / (fan_in + fan_out))), **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    return randn(fan_in, fan_out, mean=0.0, std=gain * (math.sqrt(2 / (fan_in + fan_out))), **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    return rand(fan_in, fan_out, low=-math.sqrt(6 / fan_in), high=math.sqrt(6 / fan_in), **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    return randn(fan_in, fan_out, mean=0.0, std=math.sqrt(2 / fan_in), **kwargs)
    ### END YOUR SOLUTION
