# optics_functions

This package provides functions to calculate various optics parameter from MAD-X twiss input.

## Getting Started

### Prerequisites

The package depends heavily on `tfs-pandas` and on `numpy`, so these packages need
to be installed in your python environment.

### Installing

The package is not yet deployed. To install it, run `pip install git+https://github.com/pylhc/optics_functions.git`.


## Description

This package serves as a library of functions to calculate various optics parameter such as RDTs from a MAD-X twiss input.
The twiss input should be provided as a `TfsDataFrames`. The functions then calculate the required optics parameter and
return them in a new `TfsDataFrame` or append it to the original `TfsDataFrame`.

## Functions

- so far none

## Authors

* **pyLHC/OMC-Team** - *Working Group* - [pyLHC](https://github.com/orgs/pylhc/teams/omc-team)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
