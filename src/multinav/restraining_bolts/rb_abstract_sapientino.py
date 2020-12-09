"""Restraining bolt for Abstract Sapientino."""
from flloat.semantics import PLInterpretation

from multinav.restraining_bolts.base import AbstractRB


class AbstractSapientinoRB(AbstractRB):
    """Restraining Bolt for abstract Sapientino."""

    def extract_sapientino_fluents(self, obs, action) -> PLInterpretation:
        """Extract Sapientino fluents."""
        # see AbstractSapientino.
        observation_offset = 1
        visit_action = 1

        color_id = obs - observation_offset
        if visit_action == action:
            fluents = {self.colors[color_id]}
        else:
            fluents = set()
        return PLInterpretation(fluents)
