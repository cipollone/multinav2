"""Test on environment."""

import json

from stable_baselines.common.vec_env import DummyVecEnv

from multinav.envs.ros_controls import make_ros_env
from multinav.helpers.misc import prepare_directories
from multinav.training import CustomCheckpointCallback

# TODO: make this work also for sapientino


def test(env_name, json_params):
    """Run a test on ros environment.

    :param env_name: the environment id (see ``multinav --help``)
    :param json_params: the path of the json file of parameters.
        "resume_file" should point to an existing checkpoint. The other
        parameters must be the one used for training.
    """
    # Json
    if not json_params:
        raise TypeError("You must supply the parameters of the preivous training.")
    with open(json_params) as f:
        params = json.load(f)

    # Init dirs
    resuming = bool(params["resume_file"])
    if not resuming:
        raise RuntimeError("Must be resuming to test")
    model_path, _ = prepare_directories(
        env_name="ros-stage",
        no_create=True,
    )

    # Make env
    ros_env = make_ros_env(params)

    # Callbacks
    checkpoint_callback = CustomCheckpointCallback(
        save_path=model_path,
        normalizer=None,
        save_freq=None,
        name_prefix="dqn",
    )

    # Reload model
    model, norm_env, _ = checkpoint_callback.load(path=params["resume_file"])

    # Reapply normalizer to env
    venv = DummyVecEnv([lambda: ros_env])
    norm_env.set_venv(venv)

    # Finally test
    env = norm_env
    for _ in range(100):
        obs = env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break
