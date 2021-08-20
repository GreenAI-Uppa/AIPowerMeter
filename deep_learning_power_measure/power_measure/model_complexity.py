from torchinfo import summary

def to_bytes(num: int) -> float:
    """Converts a number (assume floats, 4 bytes each) to megabytes."""
    return num * 4 / 1e6

def get_summary(model, input_size, device=None):
    """input_size (batch_size, *input_shape)"""
    print('summary',device)
    summ = summary(model, input_size=input_size, device=device)
    data = {"input_size": summ.input_size,
            "total_params":summ.total_params,
            "trainable_params": summ.trainable_params,
            "total_output": summ.total_output,
            "total_mult_adds": summ.total_mult_adds,
            "input_size": summ.total_input}
    return data
