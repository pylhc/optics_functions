from pathlib import Path

import pytest

from optics_functions.chromaticity import linear_chromaticity, chromatic_term, chromatic_beating

INPUT = Path(__file__).parent.parent / "inputs"
