from __future__ import annotations

import numpy as np

from data.replay_dataset import EncodedTransition
from rl.replay_buffer import PrioritizedReplayBuffer


def _make_transition(obs_dim: int, num_actions: int, action: int) -> EncodedTransition:
    return EncodedTransition(
        obs=np.ones(obs_dim, dtype=np.float32) * action,
        action=action,
        reward=float(action),
        next_obs=np.zeros(obs_dim, dtype=np.float32),
        terminated=False,
        truncated=False,
        action_mask=np.ones(num_actions, dtype=np.bool_),
        next_action_mask=np.ones(num_actions, dtype=np.bool_),
        discount=0.99,
        task_id="task",
        metadata={},
    )


def test_prioritized_replay_buffer_add_sample_and_update() -> None:
    buffer = PrioritizedReplayBuffer(capacity=8, obs_dim=6, num_actions=4)
    buffer.extend([_make_transition(6, 4, idx % 4) for idx in range(6)])
    batch = buffer.sample(4, beta=0.4)
    assert batch.obs.shape == (4, 6)
    assert batch.action_masks.shape == (4, 4)
    buffer.update_priorities(batch.indices, np.full(4, 2.0, dtype=np.float32))
    assert len(buffer) == 6
