"""
Constants
---------

Constants for the optics functions.
"""
import numpy as np
PI = np.pi
PI2 = 2 * np.pi
PI2I = 2j * np.pi
PLANES = ("X", "Y")
PLANE_TO_NUM = dict(X=1, Y=2)
PLANE_TO_HV = dict(X="H", Y="V")

# Columns ----------------------------------------------------------------------

ALPHA = "ALF"
BETA = "BET"
GAMMA = "GAMMA"
AMPLITUDE = 'AMP'
PHASE = 'PHASE'
PHASE_ADV = 'MU'

X, Y = PLANES

TUNE = 'Q'

# Column Pre- and Suffixes ---
REAL = 'REAL'
IMAG = 'IMAG'
