.. _ReleaseNotes:

Release notes
=============

0.6
---
Release date: July 31, 2017

* Python 3 compatibility. Following the release of python3-compatible ASE, we decided to jump on the wagon ourselves. The code should still work fine in python 2.7. (The exception is the tensorflow module, which still only lives inside python 2, unfortunately.)
* A community page has been added with resources such as the new mailing list and issue tracker.
* The default convergence parameters have been changed to energy-only training; force-training can be added by the user via the loss function.
  This makes convergence easier for new users.
* Convergence plots show maximum residuals as well as root mean-squared error.
* Parameters to make the Gaussian feature vectors are now output to the log file.
* The helper function :func:`~amp.descriptor.gaussian.make_symmetry_functions` has been added to more easily customize Gaussian fingerprint parameters.

0.5
---
Release date: February 24, 2017

The code has been significantly restructured since the previous version, in order to increase the modularity; much of the code structure has been changed since v0.4. Specific changes below:

* A parallelization scheme allowing for fast message passing with ZeroMQ.
* A simpler database format based on files, which optionally can be compressed to save diskspace.
* Incorporation of an experimental neural network model based on google's TensorFlow package. Requires TensorFlow version 0.11.0.
* Incorporation of an experimental bootstrap module for uncertainty analysis.

Permanently available at https://doi.org/10.5281/zenodo.322427

0.4
---
Release date: February 29, 2016

Corresponds to the publication of Khorshidi, A; Peterson*, AA. Amp: a modular approach to machine learning in atomistic simulations. Computer Physics Communications 207:310-324, 2016. http://dx.doi.org/10.1016/j.cpc.2016.05.010

Permanently available at https://doi.org/10.5281/zenodo.46737

0.3
---
Release date: July 13, 2015

First release under the new name "Amp" (Atomistic Machine-Learning Package/Potentials).

Permanently available at https://doi.org/10.5281/zenodo.20636


0.2
---
Release date: July 13, 2015

Last version under the name "Neural: Machine-learning for Atomistics". Future versions are named "Amp".

Available as the v0.2 tag in https://bitbucket.org/andrewpeterson/neural/commits/tag/v0.2


0.1
---
Release date: November 12, 2014

(Package name: Neural: Machine-Learning for Atomistics)

Permanently available at https://doi.org/10.5281/zenodo.12665.

First public bitbucket release: September, 2014.
