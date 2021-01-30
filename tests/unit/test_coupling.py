from pathlib import Path

import pytest

from optics_functions.coupling import closest_tune_approach, coupling_from_rdts, coupling_from_cmatrix

INPUT = Path(__file__).parent.parent / "inputs"
