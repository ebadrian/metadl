import os
from setuptools import find_packages, setup

NAME = "meta-dataset"
VERSION = "1.0"

here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

setup(
    name=NAME,
    version=about["__version__"],
    packages=find_packages(exclude=("tests",)),
    install_requires=[],

)