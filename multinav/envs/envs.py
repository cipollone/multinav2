"""Common inizialization and wrapper for environments."""

from typing import Any, Dict

from gym import Wrapper

from multinav.wrappers.utils import Renderer


class EnvMaker(Wrapper):
    """Initializer for all environments."""

    def __init__(self, env_config: Dict[str, Any]):
        """Initialize.

        :param env_config: dict of parameters for initializing the environments.
        """
        self.env_params = dict(env_config)
        self.env_name = self.env_params.pop("env")

        # Rooms env
        if self.env_name == "rooms2":
            from multinav.envs.env_abstract_rooms import make as make_room2
            self.env = make_room2(
                params=self.env_params,
                log_dir=self.env_params["logs-dir"],
            )
        elif self.env_name == "rooms1":
            from multinav.envs.env_grid_rooms import make as make_room1
            self.env = make_room1(
                params=self.env_params,
                log_dir=self.env_params["logs-dir"],
            )
            if self.env_params["render"]:
                self.env = Renderer(self.env)
        elif self.env_name == "rooms0":
            from multinav.envs.env_cont_rooms import make as make_room0
            self.env = make_room0(
                params=self.env_params,
                log_dir=self.env_params["logs-dir"],
            )
            if self.env_params["render"]:
                self.env = Renderer(self.env)
        elif self.env_name == "party0":
            from multinav.envs.env_ros_stage import make as make_party0
            self.env = make_party0(
                params=self.env_params,
                log_dir=self.env_params["logs-dir"],
            )
        else:
            raise RuntimeError("Environment not supported")

        # Gym
        super().__init__(self.env)
