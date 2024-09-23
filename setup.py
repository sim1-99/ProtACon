"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

setup.py file for sistem wide installation with setuptools.

"""
from setuptools import setup


setup(
    name='ProtACon',
    version='0.1.0',
    author='Simone Chiarella',
    author_email='simonechiarella99@gmail.com',
    packages=['ProtACon'],
    python_requires='>= 3.10',
    install_requires=[
        'biopython',
        'matplotlib',
        'numpy',
        'pandas',
        'rcsbsearchapi',
        'rich',
        'scipy',
        'seaborn',
        'torch',
        'transformers',
    ],
    entry_points={
        'console_scripts': [
            'ProtACon = ProtACon.__main__:main',
        ]
    }
)
