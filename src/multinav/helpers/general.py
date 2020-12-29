"""Python language utilities."""

import signal
from typing import Dict


class QuitWithResources:
    """Close the resources when ctrl-c is pressed."""

    __deleters: Dict = {}
    __initialized = False

    def __init__(self):
        """Don't instantiate."""
        raise TypeError("Don't instantiate this class")

    @staticmethod
    def close():
        """Close all and quit."""
        for _, deleter in QuitWithResources.__deleters.items():
            deleter()
        quit()

    @staticmethod
    def add(name, deleter):
        """Declare a new resource to be closed.

        :param name: any identifier for this resource.
        :param deleter: callable to be used when closing.
        """
        if not QuitWithResources.__initialized:
            signal.signal(signal.SIGINT, lambda _sig, _frame: QuitWithResources.close())
            QuitWithResources.__initialized = True

        if name in QuitWithResources.__deleters:
            raise ValueError("This name is already used")

        QuitWithResources.__deleters[name] = deleter

    @staticmethod
    def remove(name):
        """Remove a resource.

        :param name: identifier of a resource.
        """
        if name not in QuitWithResources.__deleters:
            raise ValueError(str(name) + " is not a resource")

        QuitWithResources.__deleters.pop(name)
