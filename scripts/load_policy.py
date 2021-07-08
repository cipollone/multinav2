"""Load the policy from pickle."""

import pickle
import sys

import numpy as np

assert len(sys.argv) == 2, "Expecting a pickle object as argument"

tabular_agent_ckpt = sys.argv[1]
with open(tabular_agent_ckpt, "rb") as f:
    q_table = pickle.load(f)

policy = {obs: np.argmax(q) for obs, q in q_table.items()}
