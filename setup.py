import pathlib

import setuptools

MODULE_NAME = "optics_functions"
# The directory containing this file
TOPLEVEL_DIR = pathlib.Path(__file__).parent.absolute()
ABOUT_FILE = TOPLEVEL_DIR / MODULE_NAME / "__init__.py"
README = TOPLEVEL_DIR / "README.md"


def about_package(init_posixpath: pathlib.Path) -> dict:
    """
    Return package information defined with dunders in __init__.py as a dictionary, when
    provided with a PosixPath to the __init__.py file.
    """
    about_text: str = init_posixpath.read_text()
    return {
        entry.split(" = ")[0]: entry.split(" = ")[1].strip('"')
        for entry in about_text.strip().split("\n")
        if entry.startswith("__")
    }


ABOUT_OPTICS_FUNCTIONS = about_package(ABOUT_FILE)

with README.open("r") as docs:
    long_description = docs.read()


# Dependencies for the package itself
DEPENDENCIES = [
    "numpy>=1.19.0",
    "pandas>=1.0",
    "tfs-pandas>=2.0"
]

# Extra dependencies
EXTRA_DEPENDENCIES = {
    "test": ["pytest>=5.2", "pytest-cov>=2.9"],
    "doc": ["sphinx", "sphinx_rtd_theme"],
}
EXTRA_DEPENDENCIES.update(
    {"all": [elem for list_ in EXTRA_DEPENDENCIES.values() for elem in list_]}
)

setuptools.setup(
    name=ABOUT_OPTICS_FUNCTIONS["__title__"],
    version=ABOUT_OPTICS_FUNCTIONS["__version__"],
    description=ABOUT_OPTICS_FUNCTIONS["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=ABOUT_OPTICS_FUNCTIONS["__author__"],
    author_email=ABOUT_OPTICS_FUNCTIONS["__author_email__"],
    url=ABOUT_OPTICS_FUNCTIONS["__url__"],
    packages=setuptools.find_packages(include=(MODULE_NAME,)),
    include_package_data=True,
    python_requires=">=3.6",
    license=ABOUT_OPTICS_FUNCTIONS["__license__"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    install_requires=DEPENDENCIES,
    tests_require=EXTRA_DEPENDENCIES["test"],
    extras_require=EXTRA_DEPENDENCIES,
)
