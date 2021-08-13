import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    requireds = f.read().splitlines()

setup(
    name="backgroundremover",
    version="0.1.8",
    description="Background remover from image and video",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nadermx/backgroundremover",
    author="Johnathan Nader",
    author_email="john@nader.mx",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="remove, background, u2net, remove background, background remover",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6, <4",
    install_requires=requireds,
    entry_points={
        "console_scripts": [
            "backgroundremover=backgroundremover.cmd.cli:main",
        ],
    },
)
