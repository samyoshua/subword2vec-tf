from setuptools import find_packages, setup

setup(
    packages=find_packages(exclude=["examples*"]),
    version="0.0.0a1",
)