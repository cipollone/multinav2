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

"""A Gym environment that controls a remote ROS instance."""

import gym
import numpy as np

from multinav.helpers.streaming import Receiver, Sender


class RosControlsEnv(gym.Env):
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
            assert len(buff) == RosControlsEnv.state_msg_len
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
            assert len(buff) == RosControlsEnv.action_msg_len

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
        self.action_sender = RosControlsEnv.ActionSender(
            msg_length=self.action_msg_len,
            port=self.actions_port,
            wait=True,
        )
        self.state_receiver = RosControlsEnv.StateReceiver(
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


class RosGoalEnv(gym.Wrapper):
    """Goals and personalizations on the ROS environment.

    Applies a reward function and end-of-episode criterion.
    This is currently used for development.
    """

    def __init__(self, time_limit):
        """Initialize.

        :param time_limit: maximum number of timesteps per episode.
        """
        # Internal environment (transition function only)
        env = RosControlsEnv()
        gym.Wrapper.__init__(self, env)

        # Params
        self._time_limit = time_limit

    def reset(self):
        """Episode reset."""
        # Params
        self._time = 0
        self._rewarded = False

        # Env
        return self.env.reset()

    def step(self, action):
        """Execute one sep."""
        # Run the internal environment
        observation, reward, done, info = self.env.step(action)

        # Compute
        self._time += 1
        reward = self.reward(observation, reward)
        done = self.is_done(observation)
        info.update({"time": self._time})

        return observation, reward, done, info

    def reward(self, observation, reward):
        """Compute the current reward."""
        # Check whether the robot is in a square (exprerimenting)
        x, y = observation[:2]
        if 5 < x < 6 and 0 < y < 1 and not self._rewarded:
            self._rewarded = True
            return 1.0
        else:
            return 0.0

    def is_done(self, observation):
        """Return whether the episode should stop."""
        # TODO: maybe something more efficient? Reach the wall or something similar
        return self._time >= self._time_limit


def interactive_test():
    """Demonstrate that connection works: just for development."""
    # NOTE: this test may not be appropriate:
    #   Some parts use ROS time which continues to go on as we wait for input.

    # Instantiate
    ros_env = RosGoalEnv(time_limit=50)

    obs = ros_env.reset()
    print("Initial state:", obs)

    # Test loop: the agent (you) chooses an action
    while True:
        inp = input("Next action ")
        if not inp:
            continue
        action = int(inp)
        if action < 0:
            obs = ros_env.reset()
            print("Initial state:", obs)
        else:
            obs, reward, done, info = ros_env.step(action)
            print(obs, reward, done, info)
