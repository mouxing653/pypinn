import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi (if you dont need delete it)
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# you need to change all these
VERSION = '1.0.1'
DESCRIPTION = 'pinn implementation using torch.'
LONG_DESCRIPTION = 'pinn implementation using torch, it is mainly used to solve some ordinary differential equations.'

setup(
    name="pypinn",
    version=VERSION,
    author="Dongpeng Han",
    author_email="3223003076@qq.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'pinn', 'pypinn','windows','mac','linux'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
