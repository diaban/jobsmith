"""`python -m jobsmith` — same as the `jobsmith` command."""
import sys

from .cli.main import main

sys.exit(main())
