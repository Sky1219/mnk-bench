"""Allow ``python -m mnkbench``."""

import sys

from .cli import main

sys.exit(main())
