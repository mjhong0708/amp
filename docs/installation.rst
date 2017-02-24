.. _install:

==================================
Installation
==================================

AMP is python-based and is designed to integrate closely with the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ (ASE).
In its most basic form, it has few requirements:

* Python, version 2.7 is recommended.
* ASE.
* NumPy.
* SciPy.

To get more features, such as parallelization in training, a few more packages are recommended:

* Pexpect (or pxssh)
* ZMQ (or PyZMQ, the python version of ØMQ).

Certain advanced modules may contain dependencies that will be noted when they are used; for example Tensorflow for the tflow module or matplotlib for the plotting modules.

Basic installation instructions follow.

----------------------------------
Install ASE
----------------------------------

We always test against the latest version (svn checkout) of ASE, but slightly older versions (>=3.9) are likely to work as well.
Follow the instructions at the `ASE <https://wiki.fysik.dtu.dk/ase/download.html>`_ website.
ASE itself depends upon python with the standard numeric and scientific packages.
Verify that you have working versions of `NumPy <http://numpy.org>`_ and `SciPy <http://scipy.org>`_.
We also recommend `matplotlib <http://matplotlib.org>`_ in order to generate plots.

----------------------------------
Get the code
----------------------------------

The latest stable release of Amp is version 0.5, which is permanently available at `https://doi.org/10.5281/zenodo.322427 <https://doi.org/10.5281/zenodo.322427>`_.
If installing version 0.5, you should follow ignore the rest of this page and follow the instructions included with the download (see docs/installation.rst or look for v0.5 on `http://amp.readthedocs.io <http://amp.readthedocs.io>`_).

We are constantly improving *Amp* and adding features, so depending on your needs it may be preferable to use the development version rather than "stable" releases.
We run daily unit tests to try to make sure that our development code works as intended.
We recommend checking out the latest version of the code via `the project's bitbucket page <https://bitbucket.org/andrewpeterson/amp/>`_.
If you use git, check out the code with::

   $ cd ~/path/to/my/codes
   $ git clone git@bitbucket.org:andrewpeterson/amp.git

where you should replace '~/path/to/my/codes' with wherever you would like the code to be located on your computer.
If you do not use git, just download the code as a zip file from the project's `download <https://bitbucket.org/andrewpeterson/amp/downloads>`_ page, and extract it into '~/path/to/my/codes'.
Please make sure that the folder '~/path/to/my/codes/amp' includes subdirectories 'amp', 'docs', 'tests', and 'tools'.

----------------------------------
Set the environment
----------------------------------

You need to let your python version know about the existence of the amp module. Add the following line to your '.bashrc'
(or other appropriate spot), with the appropriate path substituted for '~/path/to/my/codes'::

   $ export PYTHONPATH=~/path/to/my/codes/amp:$PYTHONPATH

You can check that this works by starting python and typing the below command, verifying that the location listed from
the second command is where you expect::

   >>> import amp
   >>> print(amp.__file__)

See also the section on parallel processing for any issues that arise in making the environment work with Amp in parallel.

---------------------------------------
Recommended step: Build fortran modules
---------------------------------------

Amp works in pure python, however, it will be annoyingly slow unless the associated Fortran 90 modules are compiled to speed up several parts of the code.
The compilation of the Fortran 90 code and integration with the python parts is accomplished with f2py, which is part of NumPy.
A Fortran 90 compiler will also be necessary on the system; a reasonable open-source option is GNU Fortran, or gfortran.
This compiler will generate Fortran modules (.mod).
gfortran will also be used by f2py to generate extension module fmodules.so on Linux or fmodules.pyd on Windows.
In order to prepare the extension module the following steps need to be taken:

1. Compile model Fortran subroutines inside the model and descriptor folders by::

    $ cd <installation-directory>/amp/model

    $ gfortran -c neuralnetwork.f90

    $ cd ../descriptor

    $ gfortran -c cutoffs.f90


2. Move the modules "neuralnetwork.mod" and "cutoffs.mod" created in the last step, to the parent directory by::

    $ cd ..

    $ mv model/neuralnetwork.mod .

    $ mv descriptor/cutoffs.mod .

3. Compile the model Fortran subroutines in companion with the descriptor and neuralnetwork subroutines by something like::

    $ f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90


or on a Windows machine by::

    $ f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90 --fcompiler=gnu95 --compiler=mingw32

Note that if you update your code (e.g., with 'git pull origin master') and the fortran code changes but your version of fmodules.f90 is not updated, an exception will be raised telling you to re-compile your fortran modules.

----------------------------------
Recommended step: Run the tests
----------------------------------

We include tests in the package to ensure that it still runs as intended as we continue our development; we run these
tests on the latest build every night to try to keep bugs out. It is a good idea to run these tests after you install the
package to see if your installation is working. The tests are in the folder `tests`; they are designed to run with
`nose <https://nose.readthedocs.org/>`_. If you have nose installed, run the commands below::

   $ mkdir /tmp/amptests
   $ cd /tmp/amptests
   $ nosetests ~/path/to/my/codes/amp/tests
