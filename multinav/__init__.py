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

"""Multinav project."""

# Store cwd (ray.tune can change it) 
import pathlib
starting_cwd = pathlib.Path.cwd()

# Tensorflow2
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
tf1.enable_eager_execution()

# Register custom classes
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from multinav.algorithms.policy_net import CompositeNet
from multinav.envs.envs import EnvMaker

# Custom model
ModelCatalog.register_custom_model("composite_fc", CompositeNet)

# Custom environment
register_env("<class 'multinav.envs.envs.EnvMaker'>", EnvMaker)

