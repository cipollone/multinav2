"""Value iteration implementation."""
import numpy as np

from multinav.helpers.gym import MyDiscreteEnv


class _ValueIteration:
    """Execute value iteration."""

    def __init__(
        self,
        env: MyDiscreteEnv,
        max_iterations: int = 100,
        eps: float = 1e-5,
        discount: float = 0.9,
    ):
        self.env = env
        self.eps = eps
        self.discount = discount
        self.v = np.random.random(env.nS) * eps
        self.max_iterations = max_iterations
        self.policy_stable = False

    def can_stop(self) -> bool:
        """Decide if we can stop the main GPI loop."""
        return self.policy_stable

    def __call__(self):
        """Run value iteration against a DiscreteEnv environment."""
        delta = np.inf
        iteration = 0
        while not delta < self.eps and iteration < self.max_iterations:
            delta = 0
            for s in range(len(self.v)):
                v = self.v[s]
                new_v = np.max(self._get_next_values(s))
                self.v[s] = new_v
                delta = max(delta, abs(v - new_v))
            iteration += 1
        return self.v

    def _get_next_values(self, state):
        """Get the next value, given state and action."""
        return [
            sum(
                [
                    p * (r + self.discount * self.v[sp])
                    for (p, sp, r, _done) in self.env.P[state][action]
                ]
            )
            for action in self.env.available_actions(state)
        ]


def value_iteration(*args, **kwargs):
    """Run value iteration."""
    return _ValueIteration(*args, **kwargs)()
