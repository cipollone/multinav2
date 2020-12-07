"""Restraining bolt for Abstract Sapientino."""
from flloat.semantics import PLInterpretation

from multinav.restraining_bolts.base import AbstractRB


class AbstractSapientinoRB(AbstractRB):
    """Restraining Bolt for abstract Sapientino."""

    @classmethod
    def extract_sapientino_fluents(cls, obs, action) -> PLInterpretation:
        """Extract Sapientino fluents."""
        # see AbstractSapientino.
        observation_offset = 1
        visit_action = 1

        color_id = obs - observation_offset
        if visit_action == action:
            fluents = {cls.colors[color_id]}
        else:
            fluents = set()
        return PLInterpretation(fluents)
