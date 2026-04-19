from __future__ import annotations

import numpy as np

from data.replay_dataset import EncodedTransition
from rl.rainbow import RainbowAgent, RainbowConfig
from rl.replay_buffer import PrioritizedReplayBuffer


def _transition(obs_dim: int, num_actions: int, action: int, reward: float) -> EncodedTransition:
    mask = np.zeros(num_actions, dtype=np.bool_)
    mask[:3] = True
    return EncodedTransition(
        obs=np.ones(obs_dim, dtype=np.float32) * action,
        action=action,
        reward=reward,
        next_obs=np.ones(obs_dim, dtype=np.float32) * (action + 1),
        terminated=False,
        truncated=False,
        action_mask=mask.copy(),
        next_action_mask=mask.copy(),
        discount=0.99,
        task_id="task",
        metadata={},
    )


def test_rainbow_agent_single_update_runs() -> None:
    obs_dim = 12
    num_actions = 5
    buffer = PrioritizedReplayBuffer(capacity=16, obs_dim=obs_dim, num_actions=num_actions)
    buffer.extend([_transition(obs_dim, num_actions, idx % 3, float(idx)) for idx in range(8)])
    batch = buffer.sample(4, beta=0.4)
    agent = RainbowAgent(RainbowConfig(input_dim=obs_dim, num_actions=num_actions), device="cpu")
    metrics, priorities = agent.update(batch)
    assert metrics.loss >= 0.0
    assert priorities.shape == (4,)
