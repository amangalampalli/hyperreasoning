from __future__ import annotations

import numpy as np
import torch

from env.dsl_env import ACTION_SPACE, action_name_to_id, encode_action_mask
from models.q_network import DuelingCategoricalQNetwork, QNetworkConfig


def test_encode_action_mask_sets_valid_entries() -> None:
    mask = encode_action_mask(["SELECT_CHILD_0", "TERMINATE"])
    assert mask.dtype == np.bool_
    assert mask.shape == (len(ACTION_SPACE),)
    assert mask[action_name_to_id("SELECT_CHILD_0")]
    assert mask[action_name_to_id("TERMINATE")]


def test_masked_q_values_exclude_invalid_actions() -> None:
    model = DuelingCategoricalQNetwork(QNetworkConfig(input_dim=8, num_actions=len(ACTION_SPACE)))
    obs = torch.zeros((1, 8), dtype=torch.float32)
    logits = model(obs)
    mask = torch.zeros((1, len(ACTION_SPACE)), dtype=torch.bool)
    mask[0, 3] = True
    q_values = model.masked_q_values(logits, mask)
    action = torch.argmax(q_values, dim=1).item()
    assert action == 3
