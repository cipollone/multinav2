#!/usr/bin/env python3
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

"""
This script checks that all the Python files of the repository have the copyright notice.

In particular:
- (optional) the Python shebang
- the encoding header;
- the copyright and license notices;

It is assumed the script is run from the repository root.
"""

import itertools
import re
import sys
from pathlib import Path

HEADER_REGEX_STR = r"""(#!/usr/bin/env python3
)?# -\*- coding: utf-8 -\*-
#
# Copyright 2020 Roberto Cipollone, Marco Favorito
#
# ------------------------------
#
# This file is part of multinav\.
#
# multinav is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# \(at your option\) any later version\.
#
# multinav is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE\.  See the
# GNU Lesser General Public License for more details\.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multinav\.  If not, see <https://www\.gnu\.org/licenses/>\.
#
"""
HEADER_REGEX = re.compile(HEADER_REGEX_STR)


def check_copyright(file: Path) -> bool:
    """
    Given a file, check if the header stuff is in place.

    Return True if the files has the encoding header and the copyright notice,
    optionally prefixed by the shebang. Return False otherwise.

    :param file: the file to check.
    :return True if the file is compliant with the checks, False otherwise.
    """
    content = file.read_text()
    return re.match(HEADER_REGEX, content) is not None


if __name__ == "__main__":
    exclude_files = {Path("scripts", "whitelist.py")}
    python_files = filter(
        lambda x: x not in exclude_files,
        itertools.chain(
            Path("multinav").glob("**/*.py"),
            Path("tests").glob("**/*.py"),
            Path("scripts").glob("**/*.py"),
        ),
    )

    bad_files = [filepath for filepath in python_files if not check_copyright(filepath)]

    if len(bad_files) > 0:
        print("The following files are not well formatted:")
        print("\n".join(map(str, bad_files)))
        sys.exit(1)
    else:
        print("OK")
        sys.exit(0)
