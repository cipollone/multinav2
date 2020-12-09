"""Restraining bolt for Abstract Sapientino."""
from flloat.semantics import PLInterpretation

from multinav.restraining_bolts.base import AbstractRB


class GridSapientinoRB(AbstractRB):
    """Restraining Bolt for the grid Sapientino."""

    def extract_sapientino_fluents(self, obs, action) -> PLInterpretation:
        """Extract Sapientino fluents."""
        color_offset = 1
        is_beep = obs.get("beep") > 0
        color_id = obs.get("color")
        if is_beep and color_id > 0:
            color = self.colors[color_id - color_offset]
            fluents = {color} if color in self.get_colors() else set()
        else:
            fluents = set()
        return PLInterpretation(fluents)
