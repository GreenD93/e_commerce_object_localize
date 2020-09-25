import os
import sys

from setuptools import setup, find_packages

__version__ = '0.0.1'

setup(
    name='e_commerce_object_localize',
    description='e_commerce_object_localize for python',
    version=__version__,
    install_requires=[
        'numpy'
    ],
    url='',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['object_localizer'],
)
