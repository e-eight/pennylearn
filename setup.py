# Copyright 2021 Soham Pal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

with open("pennylearn/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = ["pennylane", "scikit-learn"]

info = {
    "name": "PennyLearn",
    "version": version,
    "maintainer": "Soham Pal",
    "maintainer_email": "dssohampal@gmail.com",
    "url": "https://github.com/e-eight/pennylearn",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "entry_points": {},
    "description": "PennyLearn is a Python quantum machine learning library built on PennyLane.",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "provides": ["pennylearn"],
    "install_requires": requirements,
    "extras_require": {"kernels": ["cvxpy", "cvxopt"]},
    "package_data": {},
    "include_package_data": True,
}

classifiers = [
    "Development Status :: Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
