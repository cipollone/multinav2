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
    nb_colors = 3
    with_beep = False

    @classmethod
    def get_colors(cls) -> List[str]:
        """Get the colors."""
        return cls.colors[: cls.nb_colors]

    @classmethod
    def get_labels(cls):
        """Get the fluents."""
        return cls.get_colors() + (["bad_beep"] if cls.with_beep else [])

    @classmethod
    @abstractmethod
    def extract_sapientino_fluents(cls, obs, action) -> PLInterpretation:
        """Extract Sapientino fluents."""
        raise NotImplementedError

    @classmethod
    def make_goal(cls) -> str:
        """
        Define the goal for Sapientino.

        :return: the string associated with the goal.
        """
        labels = AbstractRB.get_colors()
        if cls.with_beep:
            empty = "!bad_beep & !" + " & !".join(labels)
        else:
            empty = "!" + " & !".join(labels)
        f = "<(" + empty + ")*;{}>tt"
        regexp = (";(" + empty + ")*;").join(labels)
        f = f.format(regexp)
        return f

    @classmethod
    def make_sapientino_goal(cls) -> TemporalGoal:
        """Make Sapientino goal."""
        s = AbstractRB.make_goal()
        logging.info(f"Computing {s}")
        return TemporalGoal(
            formula=LDLfParser()(s),
            reward=10.0,
            labels=set(AbstractRB.get_labels()),
            reward_shaping=False,
            zero_terminal_state=False,
            extract_fluents=cls.extract_sapientino_fluents,
        )
