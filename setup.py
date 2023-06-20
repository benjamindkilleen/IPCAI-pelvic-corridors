#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

setup(
    name="cortical-breach-detection",
    version="0.0.0",
    description="An Autonomous X-ray Image Acquisition and Interpretation System for Assisting Percutaneous Pelvic Fracture Fixation.",
    author="Benjamin D. Killeen",
    author_email="killeen@jhu.edu",
    url="https://github.com/benjamindkilleen/IPCAI-pelvic-corridors",
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "hydra-core",
        "omegaconf",
        "rich",
        "numpy",
        "deepdrr",
    ],
    packages=find_packages(),
)
