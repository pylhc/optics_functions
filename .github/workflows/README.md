# Continous Integration Workflows

This package implements different workflows for CI.
They are organised as follows.

### Documentation

The `documentation` workflow triggers on any push to master, builds the documentation and pushes it to the `gh-pages` branch (if the build is successful).
It runs on `ubuntu-latest` and Python version, `Python 3.9`.

### Testing Suite

Tests are ensured in the `tests` workflow, which triggers on all pushes.
It runs on a matrix of all supported operating systems (ubuntu-20.04, ubuntu-22.04, windows-latest and macos-latest) for all supported Python versions (currently `3.7`, `3.8`, `3.9` and `3.10`).

### Test Coverage

Test coverage is calculated in the `coverage` wokflow, which triggers on pushes to `master` and any push to a `pull request`.
It runs on `ubuntu-latest` & the Python version (`Python 3.9`), and reports the coverage results of the test suite to `CodeClimate`,

### Regular Testing

A `cron` workflow triggers every Monday at 3am (UTC time) and runs the full testing suite, on all available operating systems and supported Python versions.
It also runs on `Python 3.x` so that newly released Python versions that would break tests are automatically detected.

### Publishing

Publishing to `PyPI` is done through the `publish` workflow, which triggers anytime a `release` is made of the Github repository.
It builds a `wheel`, checks it, and pushes to `PyPI` if checks are successful.
