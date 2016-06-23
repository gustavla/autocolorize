#!/usr/bin/env python
from __future__ import division, print_function, absolute_import 

from setuptools import setup
import os

if os.getenv('READTHEDOCS'):
    with open('requirements_docs.txt') as f:
        required = f.read().splitlines()
else:
    with open('requirements.txt') as f:
        required = f.read().splitlines()

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Topic :: Scientific/Engineering',
]

args = dict(
    name='autocolorize',
    version='0.2.0',
    url="https://github.com/gustavla/autocolorize",
    description="Automatic colorizaton of grayscale images using Deep Learning.",
    maintainer='Gustav Larsson',
    maintainer_email='gustav.m.larsson@gmail.com',
    install_requires=required,
    scripts=['scripts/autocolorize'],
    packages=[
        'autocolorize',
        'autocolorize.res',
    ],
    package_data={'autocolorize.res': ['autocolorize.prototxt.template']},
    license='BSD',
    classifiers=CLASSIFIERS,
)

setup(**args)
