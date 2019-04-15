.. _moleculardynamics:


==================================
Using *Amp* Potentials for Molecular Dynamics
==================================

Machine-learning parameters trained in *Amp* can be used to perform fast molecular dynamics simulations, via the `Knowledge Base for Interatomic Models <https://openkim.org/>`_ (KIM).
`LAMMPS <http://www.afs.enea.it/software/lammps/doc17/html/Section_packages.html#kim>`_ recognizes *kim* as a pair style that interfaces with the KIM repository of interatomic potentials.

To build LAMMPS with the KIM package you must first install the KIM API (library) on your system.
Below are the minimal steps you need in order to install the KIM API.
After KIM API is installed, you will need to install LAMMMPS from its `github repository <https://github.com/lammps/lammps>`_.
Finally we will need to install the model driver that is provided in the *Amp* repository.
In the followings we discuss each of these steps.

In this installation instruction, we assume that the following requirements are installed on your system:

* git
* make
* cmake (If it is not installed on your system see `here <https://cmake.org/install/>`_.)
* GNU compilers (gcc, g++, gfortran) version 4.8.x or higher.


----------------------------------
Installation of KIM API
----------------------------------

You can follow the instructions given at the OpenKIM `github repository <https://github.com/openkim/kim-api/blob/master/INSTALL>`_ to install KIM API.
In short, you need to clone the repository by::

   $ git clone https://github.com/openkim/kim-api.git

Next do the following::

   $ cd kim-api-master && mkdir build && cd build
   $ FC=gfortran-4.8 cmake .. -DCMAKE_BUILD_TYPE=Release
   $ make
   $ sudo make install
   $ sudo ldconfig

The second line forces cmake to use gfortran-4.8 as the fortran compiler.
We saw gfortran-5 throws error "Error: TS 29113/TS 18508: Noninteroperable array" but gfortran-4.8 should work fine.
Now you can list model and model drivers available in KIM API by::

   $ kim-api-collections-management list

or install and remove models and model drivers, etc.
For a detailed explanation of possible options see `here <https://openkim.org/kim-api/>`_.


----------------------------------
Building LAMMPS
----------------------------------

Clone LAMMPS source files from the `github repository <https://github.com/lammps/lammps>`_::

   $ git clone https://github.com/lammps/lammps.git

Now you can do the following to build LAMMPS::

   $ cd lammps && mkdir build && cd build
   $ cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ -D CMAKE_Fortran_COMPILER=gfortran -D PKG_KIM=on -D KIM_LIBRARY=$"/usr/local/lib/libkim-api.so" -D KIM_INCLUDE_DIR=$"/usr/local/include/kim-api" ../cmake
   $ make


----------------------------------
Installation of *amp_model_driver*
----------------------------------

Now you are ready to install the *amp_model_driver* provided on this repository.
To do that first change to *amp-kim* directory by::

   $ cd /amp_directory/amp/tools/amp-kim/

where *amp_directory* is where your *Amp* source files are located.

Then make a copy of the fortran modules inside the *amp_model_driver* directory by::

   $ cp ../../amp/descriptor/gaussian.f90 amp_model_driver/gaussian.F90
   $ cp ../../amp/descriptor/cutoffs.f90 amp_model_driver/cutoffs.F90
   $ cp ../../amp/model/neuralnetwork.f90 amp_model_driver/neuralnetwork.F90

Finally you can install the *amp_model_driver* by::

   $ kim-api-collections-management install user ./amp_model_driver

You can now remove the fortran modules that you copied earlier::

   $ rm amp_model_driver/gaussian.F90
   $ rm amp_model_driver/cutoffs.F90
   $ rm amp_model_driver/neuralnetwork.F90


----------------------------------
Installation of *amp_parametrized_model*
----------------------------------

Now that you have *amp_model_driver* installed, you need to install the parameters also as the final step.
**Note that this is the only step that you need to repeat when you change the parameters of the machine-learning model.**
You should first parse all of the parameters of your *Amp* calculator to a text file by::


.. code-block:: python

 from amp import Amp
 from amp.convert import save_to_openkim
 
 calc = Amp(...)
 calc.train(...)
 save_to_openkim(calc)


where the last line parses the parameters of the calc object into a text file called *amp.params*.

You should then copy the generated text file into the *amp_parameterized_model* sub-directory of the *Amp* source directory::

   $ cp /working_directory/amp.params amp_directory/amp/tools/amp-kim/amp_parameterized_model/.

where *working_directory* is where *amp.params* is located initially, and *amp_directory* is the directory of the *Amp* source files.
Finally you change back to the *amp-kim* directory by::

   $ cd /amp_directory/amp/tools/amp-kim/

Note that installation of *amp_parameterized_model* will not work without *amp.params* being located in the */amp_directory/amp/tools/amp-kim/amp_parameterized_model* directory.
Next install your parameters by::

   $ kim-api-collections-management install user ./amp_parameterized_model

Congrats!
Now you are ready to use the *Amp* calculator with *amp.params* in you molecular dynamics simulation by an input file like this::


.. code-block:: bash

 variable	x index 1
 variable	y index 1
 variable	z index 1

 variable	xx equal 10*$x
 variable	yy equal 10*$y
 variable	zz equal 10*$z

 units		metal
 atom_style	atomic

 lattice        fcc 3.5
 region		box block 0 ${xx} 0 ${yy} 0 ${zz}
 create_box	1 box
 create_atoms	1 box
 mass		1 1.0

 velocity	all create 1.44 87287 loop geom

 pair_style      kim amp_parameterized_model
 pair_coeff	* * Pd

 neighbor	0.3 bin
 neigh_modify	delay 0 every 20 check no

 fix		1 all nve

 run		10


which, for example, is an input script for LAMMPS to do a molecular dynamics simulation of a Pd system for 10 units of time.

