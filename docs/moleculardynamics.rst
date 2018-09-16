.. _moleculardynamics:

==================================
Using *Amp* Potentials for Molecular Dynamics
==================================

Currently there are two approaches available to take the machine-learning parameters trained in *Amp* and use it to perform fast molecular dynamics simulations.
The first approach uses the `Knowledge Base for Interatomic Models <https://openkim.org/>`_ (KIM).
`LAMMPS <http://www.afs.enea.it/software/lammps/doc17/html/Section_packages.html#kim>`_ recognizes `kim` as a pair style that interfaces with the KIM repository of interatomic potentials.
To build LAMMPS with the KIM package you must first install the KIM API (library) on your system.
Below are the minimal steps you need to install the KIM API.
After KIM API is installed, you will need to install a specific version of LAMMMPS from Ryan Elliott's `repository <https://github.com/ellio167/lammps/tree/kim-v2-update>`_.
Finally we will need to install the model driver that is provided in the *Amp* repository.
In the followings we discuss each step of installation.
In this installation instruction, we assume that the following requirements are installed on your system:

* git
* cmake (If it is not installed on your system see `here` <https://cmake.org/install/>_)

----------------------------------
Installation of KIM API v2
----------------------------------

The model driver provided in this repository is in the format of KIM API version 2.
So you will need to install version 2 of the KIM API.
The `devel-v2` branch of the OpenKIM `repository <https://github.com/openkim/kim-api/tree/devel-v2>`_ works well with LAMMPS and should be downloaded::

   $ git clone -b devel-v2 https://github.com/openkim/kim-api.git

Then you can make sure that you are up-to-date with `origin/devel-v2` by::

   $ cd kim-api/

   $ git status

Next you can install KIM API v2 by simply (note you should have gone to the `kim-api` directory in the last step)::

   $ mkdir build

   $ cd build

   $ CC=gcc CXX=g++ FC=gfortran cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=${HOME}/local -DKIM_API_BUILD_MODELS_AND_DRIVERS=ON

   $ make -j4

   $ make install

Now you can make sure that all KIM tests pass by::

   $ make test

**Important step:** Now (and every time you open a new shell) you should "activate the kim-api" (set the PATH and PKG_CONFIG_PATH environment variables and load bash completions) by::

   $ source ${HOME}/local/bin/kim-api-v2-activate

Now you are ready to list model and model drivers available in KIM API v2 by::

   $ kim-api-v2-collections-management list

or install and remove models and model drivers, etc.
For a detailed explanation of possible options see `here <https://openkim.org/kim-api/>`_.


----------------------------------
Building LAMMPS
----------------------------------

Clone LAMMPS source files from the `kim-v2-update` branch of Ryan Elliott's `repository <https://github.com/ellio167/lammps/tree/kim-v2-update>`_::

   $ git clone -b kim-v2-update https://github.com/ellio167/lammps.git 

You can make sure that you are on the correct branch by::

   $ cd lammps/
   $ git status

which should tell that your branch is up-to-date with `origin/kim-v2-update`.
Next we follow the instructions given `here <https://github.com/ellio167/lammps/tree/kim-v2-update/cmake#other-packages>`_ to build LAMMPS using `cmake`.
Briefly we first make a `build` folder and then build LAMMPS inside the folder::

   $ mkdir build
   $ cd build
   $ cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ -D CMAKE_Fortran_COMPILER=gfortran -D PKG_KIM=on -D KIM_LIBRARY=$"${HOME}/local/lib/libkim-api-v2.so" -D KIM_INCLUDE_DIR=$"${HOME}/local/include/kim-api-v2" ../cmake
   $ make

