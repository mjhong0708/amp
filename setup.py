#!/usr/bin/env python

from numpy.distutils.core import Extension, setup

from amp.__version__ import version


fmodules = Extension(name='amp.fmodules',
                     sources=['amp/model/neuralnetwork.f90',
                              'amp/descriptor/gaussian.f90',
                              'amp/descriptor/cutoffs.f90',
                              'amp/descriptor/zernike.f90',
                              'amp/model.f90'])

setup(name='amp',
      version=version,
      description='Atomistic Machine-learning Package',
      long_description=open('README.md').read(),
      packages=['amp', 'amp.descriptor', 'amp.regression', 'amp.model'],
      package_dir={'amp': 'amp', 'descriptor': 'descriptor',
                   'regerressiosn': 'regression', 'model': 'model'},
      classifiers=['Programming Language :: Python',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3'],
      install_requires=['numpy>=1.7.0', 'matplotlib', 'ase'],
      ext_modules=[fmodules],
      author='Andrew Peterson',
      author_email='andrew_peterson@brown.edu'
      url='https://bitbucket.org/andrewpeterson/amp',
     )
