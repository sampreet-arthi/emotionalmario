from typing import List

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


def mask_raw_actions(
    actions: torch.Tensor, output_mappping: List, masking_val: int = 0
):
    """Converts raw interger valued actions in 0-256 range 
    to the range defined in a given mapping.

    Args:
        actions (torch.Tensor): Tensor of actions
        output_mappping (List): Mapping of actions
        masking_val (int): Value to mask invalid actions with.
            Defaults to 0 (NOOP).
    """
    device = actions.device
    byte_map = torch.zeros(256)

    valid_bytes = []
    for a, button_list in enumerate(output_mappping):
        b = 0
        for button in button_list:
            b |= button_map[button]
        byte_map[b] = a
        valid_bytes.append(b)

    mask = torch.zeros_like(actions).to(bool).to(device)
    for b in valid_bytes:
        mask |= actions == b

    masked_actions = byte_map[actions * mask + masking_val * (~mask)]
    return masked_actions.to(int)
