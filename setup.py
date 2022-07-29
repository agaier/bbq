import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="bbq",

    description="Tasty front end for PyRibs.",

    author="Adam Gaier",

    packages=find_packages(exclude=['data', 'figures', 'output', 'notebooks']),

    long_description=read('README.md'),

    version="0.1.0",

    install_requires=[
        'ribs==0.4.0',
        'dask>=2.0.0',
        "distributed>=2.0.0",
        'fire>=0.4.0',
        'humanfriendly>=10.0',
        'numpy>=1.17.0',
        'matplotlib>=3.0.0',
        'SciencePlots'
    ]
)
