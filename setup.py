"""
Setup script for Flash Experts package.
"""

from setuptools import setup, find_packages

setup(
    name="flash-experts",
    version="0.1.0",
    description="GPT-2 with Flash Attention and Mixture of Experts",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)