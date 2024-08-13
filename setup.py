"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

setup.py file for sistem wide installation with setuptools.

"""
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
