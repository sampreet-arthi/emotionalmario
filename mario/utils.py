
import torch

button_map = {
    "right": 0b10000000,
    "left": 0b01000000,
    "down": 0b00100000,
    "up": 0b00010000,
    "start": 0b00001000,
    "select": 0b00000100,
    "B": 0b00000010,
    "A": 0b00000001,
    "NOOP": 0b00000000,
}


def convert_raw_actions(actions: torch.Tensor):
    """Converts raw interger valued actions in 0-256 range 
    to the range defined in a given mapping.

    Args:
        actions (torch.Tensor): Tensor of actions
        mapping (List): Mapping of actions
    """
    return torch.randn(len(actions), 7)
