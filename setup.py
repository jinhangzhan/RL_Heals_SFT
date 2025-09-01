from setuptools import setup

from setuptools import setup, find_packages


setup(
    name="gym-cards and virl",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "Pillow",
    ],
)