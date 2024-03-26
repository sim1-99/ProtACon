#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""setup.py file for sistem wide installation with setuptools."""

__author__ = 'Simone Chiarella'
__email__ = 'simone.chiarella@studio.unibo.it'

from setuptools import setup


setup(
    name='ProtACon',
    version='0.1.0',
    packages=['ProtACon'],
    # install_requires = ["required_package", ],
    entry_points={
        'console_scripts': [
            'ProtACon = ProtACon.__main__:main',
        ]
    })
