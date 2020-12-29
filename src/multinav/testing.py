"""Test on environment."""

import json

from multinav.envs.cont_sapientino import make_sapientino_cont_env
from multinav.envs.ros_controls import make_ros_env
from multinav.helpers.misc import prepare_directories
from multinav.training import TrainStableBaselines


def test(env_name, json_params):
    """Run a test on ros environment.

    :param env_name: the environment id (see ``multinav --help``)
    :param json_params: the path of the json file of parameters.
        "resume_file" should point to an existing checkpoint. The other
        parameters must be the one used for training.
    """
    # Settings
    if not json_params:
        raise TypeError("You must supply the parameters of the preivous training.")
    with open(json_params) as f:
        params = json.load(f)

    # Init dirs
    resuming = bool(params["resume_file"])
    if not resuming:
        raise RuntimeError("Must resume from a checkpoint in order to test")
    model_path, log_path = prepare_directories(
        env_name=env_name,
        no_create=True,
    )

    # Make environment
    if env_name == "ros":
        env = make_ros_env(params=params)
        model = TrainStableBaselines(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        ).model
        tester = TestStableBaselines(env=env, model=model)
    elif env_name == "sapientino-cont":
        env = make_sapientino_cont_env(params=params)
        model = TrainStableBaselines(
            env=env,
            params=params,
            model_path=model_path,
            log_path=log_path,
        ).model
        tester = TestStableBaselines(env=env, model=model)
    else:
        raise RuntimeError("Environment not supported")

    # Start
    tester.test()


class TestStableBaselines:
    """Define the testing loop for stable_baselines."""

    def __init__(self, env, model):
        """Initialize."""
        self.env = env
        self.model = model

    def test(self):
        """Test loop."""
        for _ in range(100):
            obs = self.env.reset()
            while True:
                self.env.render()
                action, _ = self.model.predict(obs)
                obs, _, done, _ = self.env.step(action)
                if done:
                    break
        # TODO: maybe marco has some better class for experiments
