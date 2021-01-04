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

"""Most classes in multinav.helpers.general have been tested already."""

from multinav.helpers.general import ABCWithMethods, classproperty


def test_classproperty():
    """Test classproperty decorator."""

    class A:
        @classproperty
        def b(cls):
            return 3

    a = A()
    assert a.b == 3
    assert A.b == 3


class Interface(ABCWithMethods):
    """Example of an interface."""

    _abs_methods = ["this", "and_that"]


def test_ABCWithMethods():
    """Test ABCWithMethods."""

    class A:
        pass

    class B:
        def this(self):
            return True

    class C(B):
        def and_that(self):
            return False

    assert not issubclass(A, Interface)
    assert not issubclass(B, Interface)
    assert issubclass(C, Interface)
