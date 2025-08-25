"""
Module to setup project
"""

from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # <-- ESSA LINHA É ESSENCIAL
    version="0.1.0",
    description="Projeto disciplina MO412.",
    author="Yan e Juan",
    license="MIT",
)
