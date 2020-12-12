"""Helpers related to Reward shaping wrappers."""
import gym

from multinav.helpers.gym import RewardShaper


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper for reward shaping."""

    def __init__(self, env, reward_shaper: RewardShaper):
        """Initialize the Gym wrapper."""
        super().__init__(env)
        self.reward_shaper = reward_shaper

    def step(self, action):
        """Do the step."""
        state, reward, done, info = super().step(action)
        shaping_reward = self.reward_shaper.step(state, done)
        return state, reward + shaping_reward, done, info

    def reset(self, **kwargs):
        """Reset the environment."""
        result = super().reset(**kwargs)
        self.reward_shaper.reset(result)
        return result
