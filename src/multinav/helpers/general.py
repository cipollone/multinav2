"""Python language utilities."""

import signal
from abc import ABCMeta
from typing import Dict


class ABCMeta2(ABCMeta):
    """This metaclass can be used just like ABCMeta.

    It adds the possibility to declare abstract instance attributes.
    These must be assigned to instances inside the __init__ method.
    How to use:

        class C(metaclass=ABCMeta2):
            attr = AbstractAttribute()
            ...

    It is also possible to define methods and properties with that name:
        class C(metaclass=ABCMeta2):
            def attr(self):
                ...

    Note: methods of this class are not inherited by other classes' instances.
    """

    def __init__(cls, _classname, _supers, _classdict):
        """Save abstract attributes."""
        abstract = []
        for attr in dir(cls):
            if isinstance(getattr(cls, attr), AbstractAttribute):
                abstract.append(attr)
        cls.__abstract_attributes = abstract

    def __call__(cls, *args, **kwargs):
        """Intercept instance creation."""
        # Create instance
        instance = ABCMeta.__call__(cls, *args, **kwargs)

        # Check abstract
        not_defined = []
        for attr in cls.__abstract_attributes:
            if attr not in instance.__dict__:
                not_defined.append(attr)
        if not_defined:
            raise TypeError(
                cls.__name__ + ".__init__ did not define these abstract "
                "attributes:\n" + str(not_defined)
            )

        return instance


class AbstractAttribute:
    """Define an abstract attribute. See description in ABCMeta2."""


class ABC2(metaclass=ABCMeta2):
    """Abstract class through inheritance.

    Use this class just like abc.ABC.
    """


def classproperty(getter):
    """Decorate methods as class properties."""

    class StaticGetter:
        """Descriptor."""

        def __get__(self, instance, owner):
            """Return value compute by getter."""
            return getter(owner)

    return StaticGetter()


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
