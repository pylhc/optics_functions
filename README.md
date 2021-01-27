# optics_functions

[![Cron Testing](https://github.com/pylhc/optics_functions/workflows/Cron%20Testing/badge.svg)](https://github.com/pylhc/optics_functions/actions?query=workflow%3A%22Cron+Testing%22)
[![Code Climate coverage](https://img.shields.io/codeclimate/coverage/pylhc/optics_functions.svg?style=popout)](https://codeclimate.com/github/pylhc/optics_functions)
[![Code Climate maintainability (percentage)](https://img.shields.io/codeclimate/maintainability-percentage/pylhc/optics_functions.svg?style=popout)](https://codeclimate.com/github/pylhc/optics_functions)
[![GitHub last commit](https://img.shields.io/github/last-commit/pylhc/optics_functions.svg?style=popout)](https://github.com/pylhc/optics_functions)
[![GitHub release](https://img.shields.io/github/release/pylhc/optics_functions.svg?style=popout)](https://github.com/pylhc/optics_functions)

This package provides functions to calculate various optics parameters from **MAD-X TWISS** outputs.

## Getting Started

The package depends heavily on another one of our packages, `tfs-pandas`.
Installation is easily done via `pip`. The package is then used as `optics_functions`.

```
pip install optics_functions
```

Example:

```python
import optics_functions

# TODO: Include example once functionality is implemented
```

## Description

This package serves as a library of functions to calculate various optics parameters such as RDTs from a MAD-X twiss output.
The functionality mainly manipulates and returns TFS files or `TfsDataFrame` objects from the `tfs-pandas` package.

## Authors

* **pyLHC/OMC-Team** - *Working Group* - [pyLHC](https://github.com/orgs/pylhc/teams/omc-team)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.