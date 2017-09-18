#!/usr/bin/env python
# This module is the implementation of kernel ridge regression model into Amp.
#
# Author: Muammar El Khatib <muammarelkhatib@brown.edu>

import os
import numpy as np
import itertools
from collections import OrderedDict

from ase.calculators.calculator import Parameters

from . import LossFunction, Model
from ..regression import Regressor
from ..utilities import make_filename, hash_images


class KRR(Model):
    """Class implementing Kernelized Ridge Regression in Amp

    Parameters
    ----------
    kernel : str
        Choose the kernel. Default is 'rbf'.
    sigma : float
        Length scale of the Gaussian in the case of RBF. Default is 1.
    lamda : float
        Strength of the regularization in the loss when minimizing error.
    checkpoints : int
        Frequency with which to save parameter checkpoints upon training. E.g.,
        100 saves a checkpoint on each 100th training setp.  Specify None for
        no checkpoints.
    lossfunction : object
        Loss function object, if at all desired by the user.
    trainingimages : str
        PATH to Trajectory file containing the images in the training set. That
        is useful for predicting new structures.
    """
    def __init__(self, sigma=1., kernel='rbf', lamda=0., degree=3, coef0=1,
                 kernel_parwms=None, weights=None, regressor=None, mode=None,
                 trainingimages=None, version=None, fortran=False,
                 checkpoints=100, lossfunction=None):

        # Version check, particularly if restarting.
        compatibleversions = ['2015.12', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use KRR'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               'newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        np.set_printoptions(precision=30, threshold=999999999)
        p = self.parameters = Parameters()
        p.importname = '.model.kernel_ridge.KRR'
        p.version = version
        p.weights = weights
        p.mode = mode
        p.kernel = self.kernel = kernel
        p.sigma = self.sigma = sigma
        p.lamda = self.lamda = lamda
        self.trainingimages = p.trainingimages = trainingimages
        self.kernel_e = {}  # Kernel dictionary for energies
        self.kernel_f = {}  # Kernel dictionary for forces

        self.regressor = regressor
        self.parent = None  # Can hold a reference to main Amp instance.
        self.fortran = fortran
        self.checkpoints = checkpoints
        self.lossfunction = lossfunction

        if self.lossfunction is None:
            self.lossfunction = LossFunction()

    def fit(self, trainingimages, descriptor, log, parallel, only_setup=False):
        """Fit kernel ridge model using a loss function.

        Parameters
        ----------
        trainingimages : dict
            Hashed dictionary of training images.
        descriptor : object
            Class representing local atomic environment.
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        parallel: dict
            Parallel configuration dictionary. Takes the same form as in
            amp.Amp.
        """

        # Set all parameters and report to logfile.
        self._parallel = parallel
        self._log = log

        if self.regressor is None:
            self.regressor = Regressor()

        p = self.parameters
        tp = self.trainingparameters = Parameters()
        tp.trainingimages = trainingimages
        tp.descriptor = descriptor
        tp.fingerprints = tp.descriptor.fingerprints

        if p.mode is None:
            p.mode = descriptor.parameters.mode
        else:
            assert p.mode == descriptor.parameters.mode
        log('Regression in %s mode.' % p.mode)

        if len(list(self.kernel_e.keys())) == 0:
            log('Computing %s kernel.' % self.kernel)
            kij_args = dict(
                    trainingimages=tp.trainingimages,
                    fp_trainingimages=tp.fingerprints,
                    )

            # This is needed for both setting the size of parameters to
            # optimize and also to return the kernel for energies
            kij = self.get_energy_kernel(**kij_args)[0]

            if self.forcetraining is True:
                kijf_args = dict(
                    trainingimages=tp.trainingimages,
                    descriptor=tp.descriptor
                    )
                self.get_forces_kernel(**kijf_args)

        if p.weights is None:
            log('Initializing weights.')
            if p.mode == 'image-centered':
                raise NotImplementedError('Needs to be coded.')
            elif p.mode == 'atom-centered':
                size = kij.shape[-1]
                weights = OrderedDict()
                for hash in tp.trainingimages.keys():
                    imagefingerprints = tp.fingerprints[hash]
                    for element, fingerprint in imagefingerprints:
                        if element not in weights:
                            weights[element] = np.ones(size)
                p.weights = weights
        else:
            log('Initial weights already present.')

        if only_setup:
            return

        self.step = 0
        result = self.regressor.regress(model=self, log=log)
        return result  # True / False

    def get_energy_kernel(self, trainingimages=None, fp_trainingimages=None,
                          only_features=False):
        """Local method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing information about the images. Note
            that you have to hash them before passing them to this method.
        fp_trainingimages : object
            Fingerprints calculated using the trainingimages.
        only_features : bool
            If set to True, only the self.reference_features are built.

        Returns
        -------
        kij : list
            The kernel in form of a list.
        kernel_e : dictionary
            The kernel in a dictionary where keys are images' hashes.
        """
        # This creates a list containing all features in all images on the
        # training set.
        self.reference_features = []

        hashes = list(hash_images(trainingimages).keys())

        for hash in hashes:
            for element, afp in fp_trainingimages[hash]:
                afp = np.asarray(afp)
                self.reference_features.append(afp)

        kij = []

        if only_features is not True:
            for hash in hashes:
                self.kernel_e[hash] = {}
                kernel = []

                for index, (element, afp) in enumerate(
                        fp_trainingimages[hash]):
                    selfsymbol = element
                    selfindex = index
                    _kernel = self.kernel_matrix(
                            np.asarray(afp),
                            self.reference_features,
                            kernel=self.kernel
                            )
                    self.kernel_e[hash][(selfindex, selfsymbol)] = _kernel
                    kernel.append(_kernel)
                kij.append(kernel)

            kij = np.asarray(kij)

            return kij, self.kernel_e

    def get_forces_kernel(self, trainingimages=None, descriptor=None):
        """Local method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing information about the images. Note
            that you have to hash them before passing them to this method.
        descriptor : object
            Descriptor object containing the fingerprintprimes..

        Returns
        -------
        self.kernel_f : dictionary
            Dictionary containing images hashes and kernels per atom.
        """

        forces_features_x = []
        forces_features_y = []
        forces_features_z = []
        hashes = list(hash_images(trainingimages).keys())

        fingerprintprimes = descriptor.fingerprintprimes

        features = {}

        for hash in hashes:
            features[hash] = {}
            nl = descriptor.neighborlist[hash]
            image = trainingimages[hash]
            afps_prime_x = []
            afps_prime_y = []
            afps_prime_z = []

            # This loop assures that we are iterating from atom with index 0.
            for atom in image:
                selfsymbol = atom.symbol
                selfindex = atom.index
                selfneighborindices, selfneighboroffsets = nl[selfindex]

                features[hash][(selfindex, selfsymbol)] = {}

                selfneighborsymbols = [
                        image[_].symbol
                        for _ in selfneighborindices
                        ]

                fprime_sum_x,  fprime_sum_y, fprime_sum_z = 0., 0., 0.

                for component in range(3):
                    for nindex, nsymbol, noffset in zip(
                            selfneighborindices, selfneighborsymbols,
                            selfneighboroffsets):
                        # for calculating forces, summation runs over neighbor
                        # atoms of type II (within the main cell only)
                        if noffset.all() == 0:
                            key = selfindex, selfsymbol, nindex, nsymbol, \
                                  component
                            if component == 0:
                                fprime_sum_x += np.array(
                                        fingerprintprimes[hash][key])
                            elif component == 1:
                                fprime_sum_y += np.array(
                                        fingerprintprimes[hash][key])
                            else:
                                fprime_sum_z += np.array(
                                        fingerprintprimes[hash][key])

                    if component == 0:
                        afps_prime_x.append(fprime_sum_x)
                        features[hash][(
                            selfindex,
                            selfsymbol)][component] = fprime_sum_x
                    elif component == 1:
                        afps_prime_y.append(fprime_sum_y)
                        features[hash][(
                            selfindex,
                            selfsymbol)][component] = fprime_sum_y
                    else:
                        afps_prime_z.append(fprime_sum_z)
                        features[hash][(
                            selfindex,
                            selfsymbol)][component] = fprime_sum_z

            forces_features_x.append(afps_prime_x)
            forces_features_y.append(afps_prime_y)
            forces_features_z.append(afps_prime_z)

        # List containing all force features per component. Useful for
        # computing the kernels.
        self.reference_force_features = [
            list(itertools.chain.from_iterable(forces_features_x)),
            list(itertools.chain.from_iterable(forces_features_y)),
            list(itertools.chain.from_iterable(forces_features_z))
        ]

        for hash in hashes:
            image = trainingimages[hash]
            self.kernel_f[hash] = {}

            for atom in image:
                selfsymbol = atom.symbol
                selfindex = atom.index
                self.kernel_f[hash][(selfindex, selfsymbol)] = {}
                for component in range(3):
                    afp = features[hash][(selfindex, selfsymbol)][component]
                    _kernel = self.kernel_matrix(
                            afp,
                            self.reference_force_features[component],
                            kernel=self.kernel
                            )
                    self.kernel_f[hash][
                            (selfindex, selfsymbol)][component] = _kernel

        return self.kernel_f

    @property
    def forcetraining(self):
        """Returns True if forcetraining is turned on (as determined by
        examining the convergence criteria in the loss function), else
        returns False.
        """
        if self.lossfunction.parameters['force_coefficient'] is None:
            forcetraining = False
        elif self.lossfunction.parameters['force_coefficient'] > 0.:
            forcetraining = True
        return forcetraining

    @property
    def vector(self):
        """Access to get or set the model parameters (weights each kernel) as
        a single vector, useful in particular for regression.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        if self.parameters['weights'] is None:
            return None
        p = self.parameters
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights)
        return self.ravel.to_vector(weights=p.weights)

    @vector.setter
    def vector(self, vector):
        p = self.parameters

        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights)
        weights = self.ravel.to_dicts(vector)
        p['weights'] = weights

    def get_loss(self, vector):
        """Method to be called by the regression master.

        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the loss (cost) function.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """

        if self.step == 0:
            filename = make_filename(self.parent.label,
                                     '-initial-parameters.amp')
            filename = self.parent.save(filename, overwrite=True)
        if self.checkpoints:
            if self.step % self.checkpoints == 0:
                path = os.path.join(self.parent.label + '-checkpoints/')
                if self.step == 0:
                    if not os.path.exists(path):
                        os.mkdir(path)
                self._log('Saving checkpoint data.')
                filename = make_filename(path,
                                         'parameters-checkpoint-%d.amp'
                                         % self.step)
                filename = self.parent.save(filename, overwrite=True)
        loss = self.lossfunction.get_loss(vector, lossprime=False)['loss']
        if hasattr(self, 'observer'):
            self.observer(self, vector, loss)
        self.step += 1
        return loss

    def get_lossprime(self, vector):
        """Method to be called by the regression master.

        Takes one and only one input, a vector of parameters.  Returns one
        output, the value of the derivative of the loss function with respect
        to model parameters.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        return self.lossfunction.get_loss(vector,
                                          lossprime=True)['dloss_dparameters']

    @property
    def lossfunction(self):
        """Allows the user to set a custom loss function.

        For example,
        >>> from amp.model import LossFunction
        >>> lossfxn = LossFunction(energy_tol=0.0001)
        >>> calc.model.lossfunction = lossfxn

        Parameters
        ----------
        lossfunction : object
            Loss function object, if at all desired by the user.
        """
        return self._lossfunction

    @lossfunction.setter
    def lossfunction(self, lossfunction):
        if hasattr(lossfunction, 'attach_model'):
            lossfunction.attach_model(self)  # Allows access to methods.
        self._lossfunction = lossfunction

    def calculate_atomic_energy(self, afp, index, symbol, hash=None,
                                fp_trainingimages=None, trainingimages=None,
                                kernel=None, sigma=None):
        """
        Given input to the KRR model, output (which corresponds to energy)
        is calculated about the specified atom. The sum of these for all
        atoms is the total energy (in atom-centered mode).

        Parameters
        ---------
        index: int
            Index of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).
        symbol : str
            Symbol of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).
        hash : str
            hash of desired image to compute
        kernel : str
            The kernel to be computed in the case that Amp.load is used.

        Returns
        -------
        atomic_amp_energy : float
            Total energy.
        """
        if self.parameters.mode != 'atom-centered':
            raise AssertionError('calculate_atomic_energy should only be '
                                 ' called in atom-centered mode.')

        weights = self.parameters.weights

        if len(list(self.kernel_e.keys())) == 0 or hash not in self.kernel_e:
            kij_args = dict(
                    trainingimages=trainingimages,
                    fp_trainingimages=fp_trainingimages,
                    only_features=True
                    )

            # This is needed for both setting the size of parameters to
            # optimize and also to return the kernel for energies
            self.get_energy_kernel(**kij_args)
            kernel = self.kernel_matrix(
                            np.asarray(afp),
                            self.reference_features,
                            kernel=self.kernel,
                            sigma=sigma
                            )
            atomic_amp_energy = kernel.dot(weights[symbol])
        else:
            atomic_amp_energy = self.kernel_e[hash][
                    ((index, symbol))].dot(weights[symbol])
        return atomic_amp_energy

    def calculate_force(self, index, symbol, component, hash=None):
        """Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        Parameters
        ----------
        afp : list
            Atomic fingerprints in the form of a list to be used as input to
            the neural network.
        derafp : list
            Derivatives of atomic fingerprints in the form of a list to be used
            as input to the neural network.
        direction : int
            Direction of force.
        nindex : int
            Index of the neighbor atom which force is acting at.  (only used in
            the atom-centered mode)
        nsymbol : str
            Symbol of the neighbor atom which force is acting at.  (only used
            in the atom-centered mode)

        Returns
        -------
        float
            Force.
        """
        weights = self.parameters.weights
        key = index, symbol

        force = self.kernel_f[hash][key][component].dot(weights[symbol])
        force *= -1.
        return force

    def kernel_matrix(self, feature, features, kernel='rbf', sigma=1.):
        """This method takes as arguments a feature vector and a string that refers
        to the kernel type used.

        Parameters
        ----------
        features : list or numpy array
            Column vector containing the fingerprint of the atoms in images.
        kernel : str
            Select the kernel to be used. Supported kernels are: 'linear',
            rbf', 'exponential, and 'laplacian'.

        Returns
        -------
        K : array
            The kernel matrix.

        Notes
        -----
        Kernels may differ a lot between them. The kernel_matrix method has to
        contain algorithms needed to build the desired matrix. The computation
        of the kernel, is done by auxiliary functions that are located at the
        end of the KRR class.
        """
        features = np.asarray(features)
        feature = np.asarray(feature)
        K = []

        if kernel == 'linear':
            K = linear(features)

        # All kernels in this control flow share the same structure
        elif (kernel == 'rbf' or kernel == 'laplacian' or
              kernel == 'exponential'):

            for afp in features:
                K.append(rbf(feature, afp, sigma=self.sigma))

        else:
            raise NotImplementedError('This kernel needs to be coded.')

        return np.asarray(K)


"""
Auxiliary functions to compute different kernels
"""


def linear(features):
    """ Compute a linear kernel """
    linear = np.dot(features, features.T)
    return linear


def rbf(featurex, featurey, sigma=1.):
    """ Compute the rbf (AKA Gaussian) kernel.  """
    rbf = np.exp(-(np.linalg.norm(featurex - featurey)**2) / 2 * sigma**2)
    return rbf


def exponential(featurex, featurey, sigma=1.):
    """ Compute the exponential kernel"""
    exponential = np.exp(-(np.linalg.norm(featurex - featurey)) / 2 * sigma**2)
    return exponential


def laplacian(featurex, featurey, sigma=1.):
    """ Compute the laplacian kernel"""
    laplacian = np.exp(-(np.linalg.norm(featurex - featurey)) / sigma)
    return laplacian


class Raveler(object):
    """Raveler class inspired by neuralnetwork.py """
    def __init__(self, weights):
        self.count = 0
        self.weights_keys = []

        for key in weights.keys():
            self.weights_keys.append(key)
            self.count += len(weights[key])

    def to_vector(self, weights):
        vector = [weights[key] for key in self.weights_keys]
        return np.ravel(vector)

    def to_dicts(self, vector):
        """Puts the vector back into weights and scalings dictionaries of the
        form initialized. vector must have same length as the output of
        unravel."""

        assert len(vector) == self.count
        first = 0
        last = 0
        weights = OrderedDict()
        step = int(len(vector) / len(self.weights_keys))

        for k in self.weights_keys:
            if k not in weights.keys():
                last += step
                weights[k] = vector[first:last]
                first += step
        return weights
