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

----------------------------------
Install ASE
----------------------------------

We always test against the latest version (svn checkout) of ASE, but slightly older versions (>=3.9.0) are likely to work
as well. Follow the instructions at the `ASE <https://wiki.fysik.dtu.dk/ase/download.html>`_ website. ASE itself depends
upon python with the standard numeric and scientific packages. Verify that you have working versions of
`NumPy <http://numpy.org>`_ and `SciPy <http://scipy.org>`_. We also recommend `matplotlib <http://matplotlib.org>`_ in
order to generate plots.

----------------------------------
Check out the code
----------------------------------

As a relatively new project, it may be preferable to use the development version rather than "stable" releases, as improvements are constantly being made and features added.
We run daily unit tests to make sure that our development code works as intended.
We recommend checking out the latest version of the code via `the project's bitbucket
page <https://bitbucket.org/andrewpeterson/amp/>`_. If you use git, check out the code with::

   $ cd ~/path/to/my/codes
   $ git clone git@bitbucket.org:andrewpeterson/amp.git

where you should replace '~/path/to/my/codes' with wherever you would like the code to be located on your computer.
If you do not use git, just download the code as a zip file from the project's
`download <https://bitbucket.org/andrewpeterson/amp/downloads>`_ page, and extract it into '~/path/to/my/codes'. Please make sure that the folder '~/path/to/my/codes/amp' includes the script '__init__.py' as well as the folders 'descriptor', 'model', 'regression', ... 

----------------------------------
Set the environment
----------------------------------

You need to let your python version know about the existence of the amp module. Add the following line to your '.bashrc'
(or other appropriate spot), with the appropriate path substituted for '~/path/to/my/codes'::

   $ export PYTHONPATH=~/path/to/my/codes:$PYTHONPATH

You can check that this works by starting python and typing the below command, verifying that the location listed from
the second command is where you expect::

   >>> import amp
   >>> print(amp.__file__)

See also the section on parallel processing for any issues that arise in making the environment work with Amp in parallel.
 
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
