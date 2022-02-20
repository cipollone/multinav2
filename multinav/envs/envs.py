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

        # Interact env
        if self.env_name == "task2":
            from multinav.envs.env_abstract_sapientino import make as make_task2
            self.env = make_task2(
                params=self.env_params,
                log_dir=self.env_params["logs-dir"],
            )
        elif self.env_name == "task1":
            from multinav.envs.env_grid_sapientino import make as make_task1
            self.env = make_task1(
                params=self.env_params,
                log_dir=self.env_params["logs-dir"],
            )
            if self.env_params["render"]:
                self.env = Renderer(self.env)
        # Rooms env
        elif self.env_name == "rooms2":
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
        else:
            raise RuntimeError("Environment not supported")
