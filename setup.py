import pathlib
from setuptools import setup

# The directory containing this file
_this_dir = pathlib.Path(__file__).parent

# The text of the README file
long_description = (_this_dir / "README.md").read_text()

# Exec version file
exec((_this_dir / "mastodon_reader" / "version.py").read_text())


setup(
    name="Mastodon Reader",
    packages=["mastodon_reader"],
    version=__version__,
    description="Mastodon project file reader for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ist.ac.at/csommer/mastodon_reader",
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    #    entry_points = {'console_scripts': []},
    author="Christoph Sommer",
    author_email="christoph.sommer23@gmail.com",
    install_requires=["numpy", "pandas", "networkx", "matplotlib"],
)

