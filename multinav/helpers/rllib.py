"""Utilities related to rllib."""

from typing import Dict, Optional

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import EnvType, PolicyID


class DiscountedRewardLogger(DefaultCallbacks):
    """Callback that logs discounted reward metric."""

    def on_sub_environment_created(
        self,
        *,
        worker: "RolloutWorker",
        sub_environment: EnvType,
        env_context: EnvContext,
        **kwargs,
    ) -> None:
        """Run when environment is created."""
        self._gamma = env_context["gamma"]

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        **kwargs
    ) -> None:
        """Run when an episode starts."""
        # Check
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )

        # Prepare
        self._discounting = 1.0
        episode.user_data["episode_return"] = 0.0

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        """Run on each episode step."""
        # Check
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        # Accumulate
        rew = episode.last_reward_for()
        episode.user_data["episode_return"] += rew * self._discounting
        self._discounting *= self._gamma

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        """Run when an episode is done."""
        # Make sure that an episode is complete
        assert worker.policy_config["batch_mode"] == "complete_episodes"

        # Store result
        episode.custom_metrics["episode_return"] = episode.user_data["episode_return"]
