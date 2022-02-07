"""Compute the number of parameters and mac operations"""

import sys
try:
    from torchinfo import summary
    from torch import nn
except ImportError as e:
    pass

try:
    import tensorflow as tf
except ImportError as e:
    pass


def to_bytes(num: int) -> float:
    """Converts a number (assume floats, 4 bytes each) to megabytes."""
    return num * 4 / 1e6

def get_summary(model, input_size, device=None):
    """input_size (batch_size, *input_shape)"""
    data = {}
    if  'torchinfo' in sys.modules and isinstance(model, nn.Module):
        summ = summary(model, input_size=input_size, device=device)
        data = {"input_size": summ.input_size,
                "total_params":summ.total_params,
                "trainable_params": summ.trainable_params,
                "total_output": summ.total_output,
                "total_mult_adds": summ.total_mult_adds,
                "input_size": summ.total_input}
    elif 'tensorflow' in sys.modules:
        # add support to tensorflow
        return data
    return data
