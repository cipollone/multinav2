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
    _n_actions = 5

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
        # Episode vars
        self._time = 0

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

        self._time += 1

        # Execute
        self.action_sender.send(action)

        # Read observation
        observation = self.state_receiver.receive()

        # Compute reward and end
        reward = self.reward_at_state(observation)
        done = self.is_episode_done()

        # Diagnostics
        info = {"time": self._time}

        return (observation, reward, done, info)

    # TODO: the methods below are temporary placeholders used for development

    @staticmethod
    def interactive_test():
        """Demonstrate that connection works: this can be deleted."""
        # Instantiate
        ros_env = RosControlsEnv(5)  # Five actions on iocchi/StageROSGym@1bda032

        ros_env.reset()

        # Test loop: the agent (you) chooses an action
        while True:
            action = int(input("Next action "))
            if action < 0:
                ros_env.reset()
            else:
                obs, reward, done, info = ros_env.step(action)

            print(obs, reward, done, info)

    def reward_at_state(self, observation):
        """Return the reward resulting from one observation of the robot."""
        # Proportional to x because we want it to go left
        return -observation[0]

    def is_episode_done(self):
        """Check episode termination (dummy criterion)."""
        return self._time > 30
