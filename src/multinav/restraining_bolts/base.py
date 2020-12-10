"""
This module contains the abstract restraining bolts.

Classes that inherits from the abstract RB will have to
define the fluent extractor function.
"""
import logging
from abc import ABC, abstractmethod
from typing import List

from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation
from gym_sapientino.core.types import Colors
from temprl.wrapper import TemporalGoal


class AbstractRB(ABC):
    """Abstract class for Sapientino RB."""

    offset = 1
    colors = list(map(str, Colors))[offset:]

    def __init__(self, nb_colors: int, with_beep: bool = False):
        """Initialize the restraining bolt."""
        self.nb_colors = nb_colors
        self.with_beep = with_beep

    def get_colors(self) -> List[str]:
        """Get the colors."""
        return self.colors[: self.nb_colors]

    def get_labels(self):
        """Get the fluents."""
        return self.get_colors() + (["bad_beep"] if self.with_beep else [])

    @abstractmethod
    def extract_sapientino_fluents(self, obs, action) -> PLInterpretation:
        """Extract Sapientino fluents."""
        raise NotImplementedError

    def make_goal(self) -> str:
        """
        Define the goal for Sapientino.

        :return: the string associated with the goal.
        """
        labels = self.get_colors()
        if self.with_beep:
            empty = "!bad_beep & !" + " & !".join(labels)
        else:
            empty = "!" + " & !".join(labels)
        f = "<(" + empty + ")*;{}>tt"
        regexp = (";(" + empty + ")*;").join(labels)
        f = f.format(regexp)
        return f

    def make_sapientino_goal(self) -> TemporalGoal:
        """Make Sapientino goal."""
        s = self.make_goal()
        logging.info(f"Computing {s}")
        return TemporalGoal(
            formula=LDLfParser()(s),
            reward=10.0,
            labels=set(self.get_labels()),
            reward_shaping=False,
            zero_terminal_state=False,
            extract_fluents=self.extract_sapientino_fluents,
        )
