"""Test algorithm implementations."""
import numpy as np
from gym.envs.toy_text import FrozenLakeEnv
from gym.wrappers import TimeLimit

from multinav.algorithms.q_learning import q_learning
from multinav.helpers.gym import MyStatsRecorder, rollout


def test_q_learning():
    """Test q-learning."""
    env = TimeLimit(FrozenLakeEnv(is_slippery=False), max_episode_steps=100)
    Q = q_learning(env, nb_episodes=1000, alpha=0.1, eps=0.8, gamma=1.0)
    env = MyStatsRecorder(env)
    rollout(env, nb_episodes=10, policy=lambda _, _state: np.argmax(Q[_state]))
    assert np.mean(env.episode_rewards) == 1.0
