"""
Constants
*********

Constants for the optics functions.
"""
import numpy as np

PI = np.pi
PI2 = 2 * np.pi
PI2I = 2j * np.pi
PLANES = ("X", "Y")
PLANE_TO_NUM = dict(X=1, Y=2)
PLANE_TO_HV = dict(X="H", Y="V")

# Columns & Headers ------------------------------------------------------------
NAME = "NAME"
S = "S"
ALPHA = "ALF"
BETA = "BET"
GAMMA = "GAMMA"
AMPLITUDE = "AMP"
PHASE = "PHASE"
PHASE_ADV = "MU"
DISPERSION = "D"
CHROM_TERM = "CHROM"
F1010 = "F1010"
F1001 = "F1001"

X, Y = PLANES

TUNE = "Q"

# Column Pre- and Suffixes ---
REAL = "REAL"
IMAG = "IMAG"
DELTA = "DELTA"
DELTA_ORBIT = "D"  # MAD-X
MINIMUM = "MIN"

# Headers ---
CHROMATICITY = f"D{TUNE}"
LENGTH = "LENGTH"
