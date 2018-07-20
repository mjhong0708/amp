#!/usr/bin/env python

try:
    from numpy.distutils.core import Extension, setup
except ImportError:
    msg = ("Please install numpy (version 1.7.0 or greater) before installing "
           "Amp. (Amp uses numpy's installer so it can compile the fortran "
           "modules with f2py.) You should be able to do this with a command"
           " like:"
           "   $ pip install numpy")
    raise RuntimeError(msg)


# Get current version of Amp into variable 'version'.
version = open('amp/VERSION').read().strip()

# Fortran modules to be compiled by numpy's f2py.
fmodules = Extension(name='amp.fmodules',
                     sources=['amp/model/neuralnetwork.f90',
                              'amp/descriptor/gaussian.f90',
                              'amp/descriptor/cutoffs.f90',
                              'amp/descriptor/zernike.f90',
                              'amp/model.f90'])

setup(name='amp-atomistics',
      version=version,
      description='Atomistic Machine-learning Package',
      long_description=open('README').read(),
      packages=['amp', 'amp.descriptor', 'amp.regression', 'amp.model'],
      package_dir={'amp': 'amp', 'descriptor': 'descriptor',
                   'regression': 'regression', 'model': 'model'},
      classifiers=['Programming Language :: Python',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3'],
      install_requires=['numpy>=1.7.0', 'matplotlib', 'ase', 'pyzmq',
                        'pexpect'],
      ext_modules=[fmodules],
      author='Andrew Peterson',
      author_email='andrew_peterson@brown.edu',
      url='https://bitbucket.org/andrewpeterson/amp',
      package_data={'amp': ['VERSION']},
      )
