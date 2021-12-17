import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bbq",
    version="0.1.0",
    author="Adam Gaier",
    author_email="adam.gaier@autodesk.com",
    description="Helper functions for working with PyRibs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.autodesk.com/gaiera/ribs_helpers",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)