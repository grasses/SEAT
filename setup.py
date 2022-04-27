#!/usr/bin/env python

from distutils.core import setup

setup(name='SEAT',
      version='1.0.1',
      description='code for paper SEAT: Similarity Encoder by Adversarial Training for Detecting Model Extraction Attack Queries',
      author='homeway',
      author_email='homeway.me@gmail.com',
      url='https://github.com/grasses/SEAT',
      packages=['seat.seat', 'seat.trans'],
     )