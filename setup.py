import atexit
import glob
import os
import shutil

from setuptools import find_packages, setup
from setuptools.command.install import install


# --- Install style files for shart
"""
This code is based on a StackOverflow answer:
https://stackoverflow.com/questions/31559225/how-to-ship-or-distribute-a-matplotlib-stylesheet
"""

def install_styles():
    import matplotlib

    # Find all style files
    stylefiles = glob.glob('bbq/styles/*.mplstyle', recursive=True)
    # Find stylelib directory (where the *.mplstyle files go)
    mpl_stylelib_dir = os.path.join(matplotlib.get_configdir(), "stylelib")
    if not os.path.exists(mpl_stylelib_dir):
        os.makedirs(mpl_stylelib_dir)
    # Copy files over
    print("Installing styles into", mpl_stylelib_dir)
    for stylefile in stylefiles:
        print(os.path.basename(stylefile))
        shutil.copy(
            stylefile,
            os.path.join(mpl_stylelib_dir, os.path.basename(stylefile)))

class PostInstallMoveFile(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(install_styles)
# --- 

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="bbq",

    description="Tasty front end for PyRibs.",

    author="Adam Gaier",

    packages=find_packages(exclude=['data', 'figures', 'output', 'notebooks', 
                                    'experiment', 'tests']),

    long_description=read('README.md'),

    version="0.1.0",
    
    url="https://github.com/agaier/bbq",
    
    install_requires=[
        'ribs==0.4.0',
        'dask>=2.0.0',
        "distributed>=2.0.0",
        'fire>=0.4.0',
        'humanfriendly>=10.0',
        'numpy>=1.17.0',
        'matplotlib>=3.0.0',
    ],

    cmdclass={'install': PostInstallMoveFile, },
)
