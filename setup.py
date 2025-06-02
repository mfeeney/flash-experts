# setup.py
from setuptools import setup, find_packages

setup(
    name="flash-experts",
    version="0.1.0",
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    packages=find_packages(where="src"),  # Look for packages in src/
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
    ],
)