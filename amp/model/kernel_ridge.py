#!/usr/bin/env python
# This module is the implementation of kernel ridge regression model into Amp.
#
# Author: Muammar El Khatib <muammarelkhatib@brown.edu>

import os
import itertools
from collections import OrderedDict
import numpy as np
from scipy.linalg import cholesky

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
        Length scale of the Gaussian in the case of RBF, exponential, and
        laplacian kernels. Default is 1.
    lamda : float
        Strength of the regularization in the loss function when minimizing
        error.
    checkpoints : int
        Frequency with which to save parameter checkpoints upon training. E.g.,
        100 saves a checkpoint on each 100th training setp.  Specify None for
        no checkpoints. Default is None.
    lossfunction : object
        Loss function object.
    trainingimages : str
        Path to Trajectory file containing the images in the training set. This
        is useful for predicting new structures.
    cholesky : bool
        Wether or not we are using Cholesky decomposition to determine the
        weights.
    weights: dict
        Dictionary of weights.
    weights_independent : bool
        Wheter or not the weights are going to be split for energy and forces.
    """
    def __init__(self, sigma=1., kernel='rbf', lamda=0., weights=None,
                 regressor=None, mode=None, trainingimages=None, version=None,
                 fortran=False, checkpoints=None, lossfunction=None,
                 cholesky=False, weights_independent=True):

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
        self.weights_independent = p.weights_independent = weights_independent
        p.mode = mode
        p.kernel = self.kernel = kernel
        p.sigma = self.sigma = sigma
        p.lamda = self.lamda = lamda
        p.cholesky = self.cholesky = cholesky
        self.trainingimages = p.trainingimages = trainingimages
        self.kernel_e = {}  # Kernel dictionary for energies
        self.kernel_f = {}  # Kernel dictionary for forces

        self.regressor = regressor
        self.parent = None  # Can hold a reference to main Amp instance.
        self.fortran = fortran
        self.checkpoints = checkpoints
        self.lossfunction = lossfunction
        self.properties = []

        if self.lossfunction is None:
            self.lossfunction = LossFunction()

    def fit(self, trainingimages, descriptor, log, parallel, only_setup=False):
        """Fit kernel ridge model using a L2 loss function.

        Parameters
        ----------
        trainingimages : dict
            Hashed dictionary of training images.
        descriptor : object
            Class with local atomic environment information.
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

            log('Calculating %s kernel...' % self.kernel, tic='kernel')
            log('Parameters:')
            log(' lamda = %s' % self.lamda)
            log(' sigma = %s' % self.sigma)
            kij_args = dict(
                    trainingimages=tp.trainingimages,
                    fp_trainingimages=tp.fingerprints,
                    )

            # This is needed for both setting the size of parameters to
            # optimize and also to return the kernel for energies
            kij = self.get_energy_kernel(**kij_args)[0]
            self.properties.append('energy')

            if self.forcetraining is True:
                self.properties.append('forces')
                kijf_args = dict(
                    trainingimages=tp.trainingimages,
                    t_descriptor=tp.descriptor
                    )
                self.get_forces_kernel(**kijf_args)

            log('...kernel computed in', toc='kernel')

        if p.weights is None:
            log('Initializing weights.')
            if p.mode == 'image-centered':
                raise NotImplementedError('Needs to be coded.')
            elif p.mode == 'atom-centered':
                self.size = kij.shape[-1]
                weights = OrderedDict()
                for prop in self.properties:
                    weights[prop] = OrderedDict()
                    for hash in tp.trainingimages.keys():
                        imagefingerprints = tp.fingerprints[hash]
                        for element, fingerprint in imagefingerprints:
                            if (element not in weights and
                               prop is 'energy'):
                                weights[prop][element] = np.ones(self.size)
                            elif (element not in weights and
                                  prop is 'forces'):
                                if p.weights_independent is True:
                                    weights[prop][element] = np.ones(
                                            (3, self.size)
                                            )
                                else:
                                    weights[prop][element] = np.ones(self.size)
                p.weights = weights
        else:
            log('Initial weights already present.')

        if only_setup:
            return

        if self.cholesky is False:
            self.step = 0
            result = self.regressor.regress(model=self, log=log)
            return result  # True / False
        else:
            """
            This method would require to solve to systems of linear equations.
            In a atom-centered mode we don't know a priori the energy per atom
            but per image. Therefore this would work for image-centered mode.
            """
            raise NotImplementedError('Needs to be coded.')
            I = np.identity(self.size)
            K = kij.reshape(self.size, self.size)
            cholesky_U = cholesky((K + self.lamda * I))

    def get_energy_kernel(self, trainingimages=None, fp_trainingimages=None,
                          only_features=False):
        """Local method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing information about the images. Note
            that you have to hash the images before passing them to this
            method.
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

    def get_forces_kernel(self, trainingimages=None, t_descriptor=None,
                          only_features=False):
        """Local method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing the training set images. Note that
            images have to be hashed before passing them to this method.
        t_descriptor : object
            Descriptor object containing the fingerprintprimes from the
            training set.
        only_features : bool
            If set to True, only the self.reference_features are built.

        Returns
        -------
        self.kernel_f : dictionary
            Dictionary containing images hashes and kernels per atom.
        """

        forces_features_x = []
        forces_features_y = []
        forces_features_z = []

        hashes = list(hash_images(trainingimages).keys())
        fingerprintprimes = t_descriptor.fingerprintprimes

        self.force_features = {}

        for hash in hashes:
            self.force_features[hash] = {}
            image = trainingimages[hash]
            afps_prime_x = []
            afps_prime_y = []
            afps_prime_z = []

            # This loop assures that we are iterating from atom with index 0.
            for atom in image:
                selfsymbol = atom.symbol
                selfindex = atom.index
                self.force_features[hash][(selfindex, selfsymbol)] = {}

                fprime_sum_x,  fprime_sum_y, fprime_sum_z = 0., 0., 0.

                for key in fingerprintprimes[hash].keys():
                    if selfindex == key[0] and selfsymbol == key[1]:
                        if key[-1] == 0:
                            fprime_sum_x += np.array(
                                    fingerprintprimes[hash][key])
                        elif key[-1] == 1:
                            fprime_sum_y += np.array(
                                    fingerprintprimes[hash][key])
                        else:
                            fprime_sum_z += np.array(
                                    fingerprintprimes[hash][key])

                for component in range(3):
                    if component == 0:
                        afps_prime_x.append(fprime_sum_x)
                        self.force_features[hash][(
                            selfindex,
                            selfsymbol)][component] = fprime_sum_x
                    elif component == 1:
                        afps_prime_y.append(fprime_sum_y)
                        self.force_features[hash][(
                            selfindex,
                            selfsymbol)][component] = fprime_sum_y
                    else:
                        afps_prime_z.append(fprime_sum_z)
                        self.force_features[hash][(
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

        if only_features is False:
            for hash in hashes:
                image = trainingimages[hash]
                self.kernel_f[hash] = {}

                for atom in image:
                    selfsymbol = atom.symbol
                    selfindex = atom.index
                    self.kernel_f[hash][(selfindex, selfsymbol)] = {}
                    for component in range(3):
                        afp = self.force_features[hash][
                                (selfindex, selfsymbol)][component]
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
            self.ravel = Raveler(
                    p.weights,
                    weights_independent=self.weights_independent,
                    size=self.size
                    )
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
                self._log('Saving checkpoint data.')
                if self.checkpoints < 0:
                    path = os.path.join(self.parent.label + '-checkpoints')
                    if self.step == 0:
                        if not os.path.exists(path):
                            os.mkdir(path)
                    filename = os.path.join(path,
                                            '{}.amp'.format(int(self.step)))
                else:
                    filename = make_filename(self.parent.label,
                                             '-checkpoint.amp')
                self.parent.save(filename, overwrite=True)
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
        sigma : float

        Returns
        -------
        atomic_amp_energy : float
            Atomic energy on atom with index=index.
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
            atomic_amp_energy = kernel.dot(weights['energy'][symbol])
        else:
            atomic_amp_energy = self.kernel_e[hash][
                        ((index, symbol))].dot(weights['energy'][symbol])
        return atomic_amp_energy

    def calculate_force(self, index, symbol, component, fingerprintprimes=None,
                        trainingimages=None, t_descriptor=None, sigma=None,
                        hash=None):
        """Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        Parameters
        ----------
        index : integer
            Index of central atom for which the atomic force will be computed.
        symbol : str
            Symbol of central atom for which the atomic force will be computed.
        component : int
            Direction of the force.
        fingerprintprimes : list
            List of fingerprint primes.
        trainingimages : list
            Object or list containing the training set. This is needed when
            performing predictions of unseen data.
        descriptor : object
            Object containing the information about fingerprints.
        hash : str
            Unique key for the image of interest.
        sigma : float

        Returns
        -------
        force : float
            Atomic force on Atom with index=index and symbol=symbol.
        """
        weights = self.parameters.weights
        key = index, symbol

        if len(list(self.kernel_f.keys())) == 0 or hash not in self.kernel_f:
            self.get_forces_kernel(
                    trainingimages=trainingimages,
                    t_descriptor=t_descriptor,
                    only_features=True
                    )

            fprime = 0
            for afp in fingerprintprimes:
                if (index == afp[0] and symbol == afp[1] and
                   component == afp[-1]):
                    fprime += np.array(fingerprintprimes[afp])

            features = self.reference_force_features[component]
            kernel = self.kernel_matrix(
                            fprime,
                            features,
                            kernel=self.kernel,
                            sigma=sigma
                            )

            if self.weights_independent is True:
                force = kernel.dot(weights['forces'][symbol][component])
            else:
                force = kernel.dot(weights['forces'][symbol])
        else:

            if self.weights_independent is True:
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][symbol][component]
                        )
            else:
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][symbol]
                        )
        force *= -1.
        return force

    def kernel_matrix(self, feature, features, kernel='rbf', sigma=1.):
        """This method takes as arguments a feature vector and a string that refers
        to the kernel type used.

        Parameters
        ----------
        feature : list or numpy array
            Single feature.
        features : list or numpy array
            Column vector containing the fingerprints of all atoms in the
            training set.
        kernel : str
            Select the kernel to be used. Supported kernels are: 'linear',
            rbf', 'exponential, and 'laplacian'.
        sigma : float
            Length scale of the Gaussian in the case of RBF, exponential and
            laplacian kernels.

        Returns
        -------
        K : array
            The kernel matrix.

        Notes
        -----
        Kernels may differ a lot between them. The kernel_matrix method in this
        class contains algorithms to build the desired matrix. The computation
        of the kernel is done by auxiliary functions that are located at the
        end of the KRR class.
        """
        features = np.asarray(features)
        feature = np.asarray(feature)
        K = []

        call = {
                'exponential': exponential,
                'laplacian': laplacian,
                'rbf': rbf
                }

        if self.sigma is None:
            self.sigma = sigma

        if kernel == 'linear':
            for afp in features:
                K.append(linear(feature, afp))

        # All kernels in this control flow share the same structure
        elif (kernel == 'rbf' or kernel == 'laplacian' or
              kernel == 'exponential'):

            for afp in features:
                K.append(call[kernel](feature, afp, sigma=self.sigma))

        else:
            raise NotImplementedError('This kernel needs to be coded.')

        return np.asarray(K)


class Raveler(object):
    """Raveler class inspired by neuralnetwork.py

    Takes a weights dictionary created by KRR class and convert it into vector
    and back to dictionaries. This is needed for doing the optimization of the
    loss function.

    Parameters
    ----------
    weights : dict
        Dictionary containing weights per atom.
    size : int
        Number of elements in the dictionary.

    """
    def __init__(self, weights, weights_independent=None, size=None):
        self.count = 0
        self.weights_keys = []
        self.properties_keys = []
        self.size = size
        self.weights_independent = weights_independent

        for prop in weights.keys():
            self.properties_keys.append(prop)
            for key in weights[prop].keys():
                if prop is 'energy':
                    self.weights_keys.append(key)
                    self.count += len(weights[prop][key])
                elif prop is 'forces':
                    if self.weights_independent is True:
                        for component in range(3):
                            self.count += len(weights[prop][key][component])
                    else:
                        self.count += len(weights[prop][key])

    def to_vector(self, weights):
        """Convert weights dictionaries to one dimensional vectors.

        Parameters
        ----------
        weights : dict
            Dictionary of weights.

        Returns
        -------
        vector : ndarray
            One-dimensional weight vector to be used by the optimizer.
        """
        vector = []
        for prop in weights.keys():
            if prop is 'energy':
                for key in weights[prop].keys():
                    vector.append(weights[prop][key])
            elif prop is 'forces':
                if self.weights_independent is True:
                    for component in range(3):
                        for key in weights[prop].keys():
                            vector.append(weights[prop][key][component])
                else:
                    for key in weights[prop].keys():
                        vector.append(weights[prop][key])

        vector = np.ravel(vector)

        return vector

    def to_dicts(self, vector):
        """Convert vector of weights back into weights dictionaries.

        Parameters
        ----------
        vector : ndarray
            One-dimensional weight vector.

        Returns
        -------
        weights : dict
            Dictionary of weights.
        """

        assert len(vector) == self.count
        first = 0
        last = 0
        weights = OrderedDict()
        step = self.size

        for prop in self.properties_keys:
            weights[prop] = OrderedDict()
            if prop is 'energy':
                for k in self.weights_keys:
                    if k not in weights[prop].keys():
                        last += step
                        weights[prop][k] = vector[first:last]
                        first += step
            elif prop is 'forces':
                for k in self.weights_keys:
                    if (k not in weights[prop].keys() and
                            self.weights_independent is True):
                        weights[prop][k] = np.zeros((3, self.size))
                        for component in range(3):
                            last += step
                            weights[prop][k][
                                    component] = vector[first:last]
                            first += step
                    elif (k not in weights[prop].keys() and
                            self.weights_independent is False):
                        last += step
                        weights[prop][k] = vector[first:last]
                        first += step
        return weights


"""
Auxiliary functions to compute different kernels
"""


def linear(feature_i, feature_j):
    """ Compute a linear kernel """
    linear = np.dot(feature_i, feature_j)
    return linear


def rbf(feature_i, feature_j, sigma=1.):
    """ Compute the rbf (AKA Gaussian) kernel.  """
    rbf = np.exp(-(np.linalg.norm(feature_i - feature_j)**2) / 2 * sigma**2)
    return rbf


def exponential(feature_i, feature_j, sigma=1.):
    """ Compute the exponential kernel"""
    exponential = np.exp(-(np.linalg.norm(feature_i - feature_j)) /
                         2 * sigma**2)
    return exponential


def laplacian(feature_i, feature_j, sigma=1.):
    """ Compute the laplacian kernel"""
    laplacian = np.exp(-(np.linalg.norm(feature_i - feature_j)) / sigma)
    return laplacian
