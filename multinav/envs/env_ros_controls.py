# -*- coding: utf-8 -*-
#
# Copyright 2020 Roberto Cipollone, Marco Favorito
#
# ------------------------------
#
# This file is part of multinav.
#
# multinav is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multinav is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multinav.  If not, see <https://www.gnu.org/licenses/>.
#

"""A Gym environment that controls a remote ROS instance.

In this module you can just use `make()`. It defines the complete
gym.Env. `_RosControlsEnv` defines just the basic dynamics and transition
function, `_RosTerminationEnv` adds some episode termination criterion, and
_RosGoalEnv defines rewards. `make` combines them to define the environment.
"""

import gym
import numpy as np

from multinav.helpers.streaming import Receiver, Sender


class _RosControlsEnv(gym.Env):
    """Gym environment that controls ROS.

    Actions performed on this environment are forwarded to a running ROS
    instance and the states are those returned from the robot.
    This environment communicates with a running instance of
    https://github.com/iocchi/StageROSGym. See that repo to setup also the
    other side.
    """

    # NOTE: many settings depend on the communication protocol.
    #   Make sure that the other side respects the same actions, signals,
    #   and observation space.

    # Number of actions
    _n_actions = 6

    # Other (non-RL) signals
    _signals = {
        "reset": -1,
    }

    # Communication protocol
    actions_port = 30005
    states_port = 30006
    state_msg_len = 20  # a numpy vector of 5 float32
    action_msg_len = 4  # a positive numpy scalar of type int32

    class StateReceiver(Receiver):
        """Just a wrapper that deserializes state vectors."""

        def receive(self):
            """Return a state vector received.

            :return: a numpy vector of the most recent state.
            """
            # Receive
            buff = Receiver.receive(self, wait=True)

            # Deserialize
            assert len(buff) == _RosControlsEnv.state_msg_len
            array = np.frombuffer(buff, dtype=np.float32)
            return array

    class ActionSender(Sender):
        """Just a wrapper that serializes actions."""

        def send(self, action):
            """Send an action.

            :param action: a scalar action or signal.
            """
            # Serialize
            buff = np.array(action, dtype=np.int32).tobytes()
            assert len(buff) == _RosControlsEnv.action_msg_len

            # Send
            Sender.send(self, buff)

    def __init__(self):
        """Initialize."""
        # Define spaces
        self.action_space = gym.spaces.Discrete(self._n_actions)
        self.observation_space = gym.spaces.Box(
            low=float("-inf"), high=float("inf"), shape=[5], dtype=np.float32
        )
        # NOTE: Actually the third is an angle

        # Initialize connections
        self.action_sender = _RosControlsEnv.ActionSender(
            msg_length=self.action_msg_len,
            port=self.actions_port,
            wait=True,
        )
        self.state_receiver = _RosControlsEnv.StateReceiver(
            msg_length=self.state_msg_len,
            ip="localhost",
            port=self.states_port,
            wait=True,
        )

        # Connect now
        self.action_sender.start()
        print("> Serving actions on", self.action_sender.server.server_address)
        print(
            "> Connecting to ",
            self.state_receiver.ip,
            ":",
            self.state_receiver.port,
            " for states. (pause)",
            sep="",
            end=" ",
        )
        input()
        self.state_receiver.start()

    def reset(self):
        """Reset the environment to the initial state.

        :return: The initial observation.
        """
        # Send signal
        self.action_sender.send(self._signals["reset"])

        # Return initial observation
        observation = self.state_receiver.receive()

        return observation

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        :param action: the action to perform.
        :return:
            observation: the next observation
            reward (float): the scalar reward
            done (bool): whether the episode has ended
            info (dict): other infos
        """
        # Check
        if not 0 <= action < self.action_space.n:
            raise RuntimeError(str(action) + " is not an action.")

        # Execute
        self.action_sender.send(action)

        # Read observation
        observation = self.state_receiver.receive()

        return (observation, 0.0, False, {})

    def render(self, mode="human"):
        """Render the environment.

        Does nothing. Run with stage gui if you want to visualize.
        """
        pass

    @staticmethod
    def linear_velociy_in_obs(observation):
        """Return the linear velocity from an observation of this env."""
        return observation[3]


class _RosTerminationEnv(gym.Wrapper):
    """Assign a criterion for episode termination."""

    def __init__(self, env, time_limit, notmoving_limit=12):
        """Initialize.

        :param env: internal environment.
        :param time_limit: maximum number of timesteps per episode.
        :param notmoving_limit: maximum number of timesteps that the agent
            can stand still.
        """
        gym.Wrapper.__init__(self, env=env)

        # Params
        self._time_limit = time_limit
        self._not_moving_limit = notmoving_limit
        self._not_moving_time = 0

    def reset(self):
        """Episode reset."""
        self._time = 0
        self._not_moving_time = 0
        return self.env.reset()

    def step(self, action):
        """Execute one step."""
        # Run the internal environment
        observation, reward, done, info = self.env.step(action)

        # Compute
        self._time += 1
        done = self.is_done(observation)
        info.update({"time": self._time})

        return observation, reward, done, info

    def is_done(self, observation):
        """Return whether the episode should stop."""
        # Terminate if over time
        time_limited = self._time >= self._time_limit

        # Terminate if not moving for too long
        linear_vel = _RosControlsEnv.linear_velociy_in_obs(observation)
        if linear_vel < 0.05:
            self._not_moving_time += 1
        else:
            self._not_moving_time = 0
        notmoving_limited = self._not_moving_time >= self._not_moving_limit

        return time_limited or notmoving_limited


class _RosGoalEnv(gym.Wrapper):
    """Assign a goal to the ROS environment.

    Choose a reward for the ROS experiment.
    """

    def reset(self):
        """Episode reset."""
        # Params
        self._rewarded = False

        # Env
        return self.env.reset()

    def step(self, action):
        """Execute one sep."""
        # Run the internal environment
        observation, reward, done, info = self.env.step(action)

        # Compute
        reward = self.reward(observation, reward)

        return observation, reward, done, info

    def reward(self, observation, reward):
        """Compute the current reward."""
        # Check whether the robot is in a square (exprerimenting)
        x, y = observation[:2]
        if 10 < x < 11 and 0 < y < 1 and not self._rewarded:
            self._rewarded = True
            return 1.0
        else:
            return 0.0


def make(params):
    """Make the "ROS" environment.

    See the docs of the other classes in this module for futher information.
    :param params: a dictionary of parameters; see in this function the
        only ones that are used.
    :return: an object that respects the gym.Env interface.
    """
    input_env = _RosGoalEnv(
        env=_RosTerminationEnv(
            env=_RosControlsEnv(),
            time_limit=params["episode_time_limit"],
            notmoving_limit=params["notmoving_limit"],
        )
    )
    # TODO: do the same as the other environments
    return input_env
