"""Test on environment."""

import json

from stable_baselines.common.vec_env import DummyVecEnv

from multinav.envs.ros_controls import RosControlsEnv, RosGoalEnv, RosTerminationEnv
from multinav.helpers.misc import prepare_directories
from multinav.training import CustomCheckpointCallback


def test_on_ros(json_args):
    """Run a test on ros environment.

    :param json_args: the path of the json file of arguments.
        "resume_file" should point to an existing checkpoint. The other
        parameters must be the one used for training.
    """
    with open(json_args) as f:
        learning_params = json.load(f)

    # Init dirs
    resuming = bool(learning_params["resume_file"])
    if not resuming:
        raise RuntimeError("Must be resuming to test")
    model_path, _ = prepare_directories(
        env_name="ros-stage",
        no_create=True,
    )

    # Make env
    ros_env = RosGoalEnv(
        env=RosTerminationEnv(
            env=RosControlsEnv(),
            time_limit=learning_params["episode_time_limit"],
            notmoving_limit=learning_params["notmoving_limit"],
        )
    )

    # Callbacks
    checkpoint_callback = CustomCheckpointCallback(
        save_path=model_path,
        normalizer=None,
        save_freq=None,
        name_prefix="dqn",
    )

    # Reload model
    model, norm_env, _ = checkpoint_callback.load(
        path=learning_params["resume_file"],
    )

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
