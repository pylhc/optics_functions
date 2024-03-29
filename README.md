# optics_functions

[![Cron Testing](https://github.com/pylhc/optics_functions/workflows/Cron%20Testing/badge.svg)](https://github.com/pylhc/optics_functions/actions?query=workflow%3A%22Cron+Testing%22)
[![Code Climate coverage](https://img.shields.io/codeclimate/coverage/pylhc/optics_functions.svg?style=popout)](https://codeclimate.com/github/pylhc/optics_functions)
[![Code Climate maintainability (percentage)](https://img.shields.io/codeclimate/maintainability-percentage/pylhc/optics_functions.svg?style=popout)](https://codeclimate.com/github/pylhc/optics_functions)
[![GitHub last commit](https://img.shields.io/github/last-commit/pylhc/optics_functions.svg?style=popout)](https://github.com/pylhc/optics_functions)
<!-- [![GitHub release](https://img.shields.io/github/release/pylhc/optics_functions.svg?style=popout)](https://github.com/pylhc/optics_functions) -->
[![PyPI Version](https://img.shields.io/pypi/v/optics_functions?label=PyPI&logo=pypi)](https://pypi.org/project/optics_functions/)
[![GitHub release](https://img.shields.io/github/v/release/pylhc/optics_functions?logo=github)](https://github.com/pylhc/optics_functions/)
[![Conda-forge Version](https://img.shields.io/conda/vn/conda-forge/optics_functions?color=orange&logo=anaconda)](https://anaconda.org/conda-forge/optics_functions)
[![DOI](https://zenodo.org/badge/215268186.svg)](https://zenodo.org/badge/latestdoi/215268186)

This package provides functions to calculate various optics parameters from **MAD-X TWISS** outputs, such as RDTs and coupling.
The functionality mainly manipulates and returns **TFS** files or `TfsDataFrame` objects from our `tfs-pandas` package.

See the [API documentation](https://pylhc.github.io/optics_functions/) for details.

## Installing

Installation is easily done via `pip`:
```bash
python -m pip install optics_functions
```

One can also install in a `conda` environment via the `conda-forge` channel with:
```bash
conda install -c conda-forge optics_functions
```

## Example Usage

> **Warning:** In certain scenarios, e.g. in case of non-zero closed orbit, the `RDT` calculations can be unreliable for **thick** lattices.
> Convert to a _thin_ lattice by slicing the lattice to reduce the error of the analytical approximation.

#### Coupling Example:

```python
import logging
import sys

import tfs  # tfs-pandas

from optics_functions.coupling import coupling_via_cmatrix, closest_tune_approach
from optics_functions.utils import split_complex_columns

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

# read MAD-X twiss output
df_twiss = tfs.read("twiss.tfs", index="NAME")

# calculate coupling from the cmatrix
df_coupling = coupling_via_cmatrix(df_twiss)

# Example:
# print(df_coupling) 
#
#                            F1001               F1010  ...       C22     GAMMA
# NAME                                                  ...
# IP3          -0.000000+0.000004j -0.004026+0.003574j  ... -0.007140  1.000058
# MCBWV.4R3.B1  0.000001+0.000004j -0.002429+0.004805j  ... -0.009601  1.000058
# BPMW.4R3.B1   0.000001+0.000004j -0.002351+0.004843j  ... -0.009678  1.000058
# MQWA.A4R3.B1  0.000001+0.000004j -0.001852+0.005055j  ... -0.010102  1.000058
# MQWA.B4R3.B1  0.000001+0.000004j -0.001231+0.005241j  ... -0.010474  1.000058
# ...                          ...                 ...  ...       ...       ...
# MQWB.4L3.B1  -0.000000+0.000004j -0.005059+0.001842j  ... -0.003675  1.000058
# MQWA.B4L3.B1 -0.000000+0.000004j -0.004958+0.002098j  ... -0.004187  1.000058
# MQWA.A4L3.B1 -0.000000+0.000004j -0.004850+0.002337j  ... -0.004666  1.000058
# BPMW.4L3.B1  -0.000000+0.000004j -0.004831+0.002376j  ... -0.004743  1.000058
# MCBWH.4L3.B1 -0.000000+0.000004j -0.004691+0.002641j  ... -0.005274  1.000058


# calculate the closest tune approach from the complex rdts
df_dqmin = closest_tune_approach(
    df_coupling, qx=df_twiss.Q1, qy=df_twiss.Q2, method='calaga'
)

# Example:
# print(df_dqmin) 
#
#                  DELTAQMIN
# NAME
# IP3           1.760865e-07
# MCBWV.4R3.B1  1.760865e-07
# BPMW.4R3.B1   1.760866e-07
# MQWA.A4R3.B1  1.760865e-07
# MQWA.B4R3.B1  1.760865e-07
# ...                    ...
# MQWB.4L3.B1   1.760865e-07
# MQWA.B4L3.B1  1.760865e-07
# MQWA.A4L3.B1  1.760866e-07
# BPMW.4L3.B1   1.760865e-07
# MCBWH.4L3.B1  1.760865e-07

# do something with the data.
# (...)

# write out
# as the writer can only handle real data, 
# you need to split the rdts into real and imaginary parts before writing
tfs.write(
    "coupling.tfs",
    split_complex_columns(df_coupling, columns=["F1001", "F1010"]),
    save_index="NAME",
)
```

#### RDT Example:

```python
import logging
import sys

import tfs  # tfs-pandas

from optics_functions.rdt import calculate_rdts, generator, jklm2str
from optics_functions.utils import prepare_twiss_dataframe, split_complex_columns

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

# read MAD-X twiss output
df_twiss = tfs.read("twiss.tfs", index="NAME")

# generate all valid RDT names, here for RDTs of order 2
rdts = [jklm2str(*jklm) for jklm in generator(orders=[2])[2]]

# check correct signs (i.e if beam==4), merge twiss and errors, 
# add empty K(S)L columns if needed
df_twiss = prepare_twiss_dataframe(df_twiss=df_twiss, df_errors=None, max_order=5)

# do the actual rdt calculation
df_rdts = calculate_rdts(
    df_twiss,
    rdts=rdts,
    loop_phases=True,  # loop over phase-advance calculation, slower but saves memory
    feeddown=2,  # include feed-down up to this order
    complex_columns=True,  # complex output
)

# Example: 
# print(df_rdts) 
#                            F0002  ...               F2000
# NAME                              ...
# IP3           2.673376-1.045712j  ... -2.863617-0.789910j
# MCBWV.4R3.B1  2.475684-1.453081j  ... -1.927365-2.260426j
# BPMW.4R3.B1   2.470411-1.462027j  ... -1.862287-2.314336j
# MQWA.A4R3.B1  2.440763-1.511004j  ... -1.413706-2.612603j
# MQWA.B4R3.B1  2.228282-1.555324j  ... -0.788608-2.855177j
# ...                          ...  ...                 ...
# MQWB.4L3.B1   2.733194+0.167312j  ... -2.632290+0.135418j
# MQWA.B4L3.B1  2.763986-0.041253j  ... -2.713212+0.063256j
# MQWA.A4L3.B1  2.804960-0.235493j  ... -2.847616-0.017922j
# BPMW.4L3.B1   2.858218-0.266543j  ... -2.970384-0.032890j
# MCBWH.4L3.B1  2.831426-0.472735j  ... -2.966818-0.149180j

# do something with the rdts.
# (...)

# write out
# as the writer can only handle real data, either set real = True above 
# or split the rdts into real and imaginary parts before writing
tfs.write(
    "rdts.tfs",
    split_complex_columns(df_rdts, columns=rdts),
    save_index="NAME"
)
```

#### Appending Example:

```python
import logging
import sys

import tfs  # tfs-pandas

from optics_functions.coupling import coupling_via_cmatrix, closest_tune_approach
from optics_functions.utils import split_complex_columns

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

# read MAD-X twiss output
df_twiss = tfs.read("twiss.tfs", index="NAME")

# calculate coupling from the cmatrix and append to original dataframe
# output=['rdts'] is used to avoid the output of the gamma and C## columns.
df_twiss[["F1001", "F1010"]] = coupling_via_cmatrix(df_twiss, output=['rdts'])

# Example:
# print(df_twiss)
# 
# Headers:
# NAME: TWISS
# TYPE: TWISS
# SEQUENCE: LHCB1
# ...
# ORIGIN: 5.05.02 Linux 64
# DATE: 01/02/21
# TIME: 19.58.08
# 
#                  KEYWORD           S  ...               F1001               F1010
# NAME                                  ...
# IP3               MARKER      0.0000  ... -0.000000+0.000004j -0.004026+0.003574j
# MCBWV.4R3.B1     VKICKER     21.8800  ...  0.000001+0.000004j -0.002429+0.004805j
# BPMW.4R3.B1      MONITOR     22.5205  ...  0.000001+0.000004j -0.002351+0.004843j
# MQWA.A4R3.B1  QUADRUPOLE     26.1890  ...  0.000001+0.000004j -0.001852+0.005055j
# MQWA.B4R3.B1  QUADRUPOLE     29.9890  ...  0.000001+0.000004j -0.001231+0.005241j
# ...                  ...         ...  ...                 ...                 ...
# MQWB.4L3.B1   QUADRUPOLE  26628.2022  ... -0.000000+0.000004j -0.005059+0.001842j
# MQWA.B4L3.B1  QUADRUPOLE  26632.0022  ... -0.000000+0.000004j -0.004958+0.002098j
# MQWA.A4L3.B1  QUADRUPOLE  26635.8022  ... -0.000000+0.000004j -0.004850+0.002337j
# BPMW.4L3.B1      MONITOR  26636.4387  ... -0.000000+0.000004j -0.004831+0.002376j
# MCBWH.4L3.B1     HKICKER  26641.0332  ... -0.000000+0.000004j -0.004691+0.002641j
```
### Modules

- `coupling` - Functions to estimate coupling from twiss dataframes and
  different methods to calculate the closest tune approach from
  the calculated coupling RDTs.
  ([**coupling.py**](optics_functions/coupling.py), [**doc**](https://pylhc.github.io/optics_functions/modules/coupling.html))
- `rdt` - Functions for the calculations of Resonance Driving Terms, as well as
  getting lists of valid driving term indices for certain orders. 
  ([**rdt.py**](optics_functions/rdt.py), [**doc**](https://pylhc.github.io/optics_functions/modules/rdt.html))
- `utils` - Helper functions to prepare the twiss dataframes for use with the optics
  functions as well as reusable utilities,
  that are needed within multiple optics calculations.
  ([**utils.py**](optics_functions/utils.py), [**doc**](https://pylhc.github.io/optics_functions/modules/utils.html))

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
