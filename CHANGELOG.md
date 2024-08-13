# optics_functions Changelog

## Version 0.1.3

- Fixed use of `np.NaN` to ensure compatibility with `numpy 2.0`.

## Version 0.1.2

- Fixed:
  - an issue that could lead to an indexing error in the closest tune approach calculation for some of the available methods.

## Version 0.1.1

A patch for some closest tune approach methods, and the addition of new methods.

- Fixed:
  - Closest tune approach calculation now properly checks for the resonance relation between f1001 and f1010 resonance driving terms.
  - Closest tune approach calculation now properly takes into account weights from element lengths

- Added:
  - New methods for closest tune approach calculation: `teapot` (equivalent to `calaga`), `teapot_franchi`, `hoydalsvik` and `hoydalsvik_alt`. References can be found in the documentation.
  - Additional tests and checks.

## Version 0.1.0

- Added:
  - Coupling calculations
  - RDT calculations
  - Utilities
  - Tests for coupling, rdt and utils
  - Documentation
  - Workflows