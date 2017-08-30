#!/usr/bin/env python
# This module is the implementation of kernel ridge regression model into Amp.
#
# Author: Muammar El Khatib <muammarelkhatib@brown.edu>

import os
import numpy as np
import itertools

from scipy.optimize import fmin
from scipy.spatial.distance import pdist, squareform

from ase.calculators.calculator import Parameters
from ase.io import Trajectory

from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

from . import LossFunction, Model
from ..regression import Regressor
from ..utilities import ConvergenceOccurred, make_filename, hash_images

class KRR(Model):
    """Class implementing Kernelized Ridge Regression

    Parameters
    ----------
    kernel : str
        Choose the kernel. Default is 'linear'.
    sigma : float
        Length scale of the Gaussian in the case of RBF. Default is 1.
    lamda : float
        Strength of the regularization in the loss function.
    checkpoints : int
        Frequency with which to save parameter checkpoints upon training. E.g.,
        100 saves a checkpoint on each 100th training setp.  Specify None for no
        checkpoints.
    lossfunction : object
        Loss function object, if at all desired by the user.
    trainingimages : str
        PATH to Trajectory file containing the images in the training set. That
        is useful for predicting new structures.
    """
    def __init__(self, sigma=1., kernel='linear', lamda=0., degree=3, coef0=1,
            kernel_parwms=None, weights=None, regressor=None, mode=None,
            trainingimages=None, version=None, fortran=False, checkpoints=100,
            lossfunction=None):

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

        self._losses = []

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
                    descriptor = tp.descriptor
                    )
                self.get_forces_kernel(**kijf_args)
                #print(tp.descriptor.neighborlist['4cfa45109444691ee4864f916b7aa4e7'])

        if p.weights is None:
            log('Initializing weights.')
            if p.mode == 'image-centered':
                raise NotImplementedError('Needs to be coded.')
            elif p.mode == 'atom-centered':
                size = kij.shape[-1]
                weights = np.ones(size)
                p.weights = weights
        else:
            log('Initial weights already present.')

        if only_setup:
            return

        self.step = 0
        result = self.regressor.regress(model=self, log=log)
        return result  # True / False

    def get_energy_kernel(self, trainingimages=None, fp_trainingimages=None):
        """Local method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing information about the images. Note
            that you have to hash them before passing them to this method.
        fp_trainingimages : object
            Fingerprints calculated using the trainingimages.
        """

        energy_features = []
        hashes = []
        features = []

        for hash in hash_images(trainingimages).keys():
            afps = []
            hashes.append(hash)

            for element, afp in fp_trainingimages[hash]:
                afp = np.asarray(afp)
                afps.append(afp)

            energy_features.append(afps)

        print(hashes)
        kij = []

        self.reference_features = list(itertools.chain.from_iterable(energy_features))

        for index, _ in enumerate(energy_features):
            kernel = []
            for atom, afp in enumerate(_):
                _kernel = self.kernel_matrix(afp, self.reference_features, kernel=self.kernel)
                kernel.append(_kernel)

            self.kernel_e[hashes[index]] = kernel
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
        fpp_trainingimages : object
            Fingerprints primes calculated using the trainingimages.
        """

        #print('I will do something.')
        forces_features = []
        hashes = []
        features = []
        fingerprintprimes = descriptor.fingerprintprimes

        for hash in hash_images(trainingimages).keys():
            print('hash = {}' .format(hash))
            #print(fingerprintprimes[hash])
            afps = []
            hashes.append(hash)
            nl = descriptor.neighborlist[hash]
            image = trainingimages[hash]

            for atom in image:
                selfsymbol = atom.symbol
                selfindex = atom.index
                selfneighborindices, selfneighboroffsets = nl[selfindex]

                selfneighborsymbols = [image[_].symbol for _ in selfneighborindices]

                #selfneighborpositions = [image.positions[_index] +
                #                         np.dot(_offset, image.get_cell())
                #                         for _index, _offset
                #                         in zip(selfneighborindices,
                #                                selfneighboroffsets)]

                for i in range(3):
                    # Calculating derivative of fingerprints of self atom w.r.t.
                    # coordinates of itself.
                    #fpprime = self.get_fingerprintprime(
                    #    selfindex, selfsymbol,
                    #    selfneighborindices,
                    #    selfneighborsymbols,
                    #    selfneighborpositions, selfindex, i)

                    #fingerprintprimes[
                    #    (selfindex, selfsymbol, selfindex, selfsymbol, i)] = \
                    #    fpprime
                    # Calculating derivative of fingerprints of neighbor atom
                    # w.r.t. coordinates of self atom.
                    fprime_sum = 0.
                    for nindex, nsymbol, noffset in zip(selfneighborindices, selfneighborsymbols, selfneighboroffsets):
                        # for calculating forces, summation runs over neighbor
                        # atoms of type II (within the main cell only)
                        if noffset.all() == 0:
                            #nneighborindices, nneighboroffsets = nl[nindex]
                            #nneighborsymbols = [image[_].symbol for _ in nneighborindices]

                            #neighborpositions = [image.positions[_index] +
                            #                     np.dot(_offset, image.get_cell())
                            #                     for _index, _offset
                            #                     in zip(nneighborindices,
                            #                            nneighboroffsets)]

                            ## for calculating derivatives of fingerprints,
                            ## summation runs over neighboring atoms of type
                            ## I (either inside or outside the main cell)
                            #fpprime = self.get_fingerprintprime(
                            #    nindex, nsymbol,
                            #    nneighborindices,
                            #    nneighborsymbols,
                            #    neighborpositions, selfindex, i)

                            #fingerprintprimes[
                            #    (selfindex, selfsymbol, nindex, nsymbol, i)] = \
                            #    fpprime
                            key = selfindex, selfsymbol, nindex, nsymbol, i
                            #print('key: %s, %s' % (key, fingerprintprimes[hash][key]))
                            fprime = np.array(fingerprintprimes[hash][key])
                            fprime_sum += fprime
                    print('component {}' .format(i))
                    print(fprime_sum)

        print(hashes)
        """
        for hash in hash_images(self.trainingimages).keys():
            afps = []
            hashes.append(hash)

            print(hash)
            #print(fpp_trainingimages[hash].keys())

            for element, afp in fpp_trainingimages[hash]:
                print(afp)
                afp = np.asarray(afp)
                afps.append(afp)

            energy_features.append(afps)


        kij = []

        self.reference_features = list(itertools.chain.from_iterable(energy_features))

        for index, _ in enumerate(energy_features):
            kernel = []
            for atom, afp in enumerate(_):
                _kernel = self.kernel_matrix(afp, self.reference_features, kernel=self.kernel)
                kernel.append(_kernel)

            self.kernel_e[hashes[index]] = kernel
            kij.append(kernel)

        kij = np.asarray(kij)

        return kij, self.kernel_e
        """

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
        """Access to get or set the model parameters as a single vector, useful
        in particular for regression.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        if self.parameters['weights'] is None:
            return None
        p = self.parameters
        vector = p.weights
        return vector

    @vector.setter
    def vector(self, vector):
        p = self.parameters
        p['weights'] = vector

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

    def calculate_atomic_energy(self, index, symbol, fingerprints, hash=None,
            fp_trainingimages=None, trainingimages=None, kernel=None, sigma=None):
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
            Energy.
        """
        if self.parameters.mode != 'atom-centered':
            raise AssertionError('calculate_atomic_energy should only be '
                                 ' called in atom-centered mode.')

        weights = self.parameters.weights

        if len(list(self.kernel_e.keys())) == 0 or hash not in self.kernel_e:
            kernel = self.get_energy_kernel(
                    trainingimages=trainingimages,
                    fp_trainingimages=fp_trainingimages
                    )[1].values()
            afp = np.asarray(fingerprints[index][1])
            kernel = self.kernel_matrix(afp, self.reference_features, kernel=self.kernel)
            #atomic_amp_energy = kernel.dot(weights)
            atomic_amp_energy = sum(kernel.dot(weights))
        else:
            atomic_amp_energy = sum(self.kernel_e[hash][index].dot(weights))
        #return asscalar(atomic_amp_energy)
        return atomic_amp_energy

    def calculate_force(self, afp, derafp,
                        direction,
                        nindex=None, nsymbol=None,):
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

        #print('I will do something one day')
        # force is multiplied by -1, because it is -dE/dx and not dE/dx.
        force = 1
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

        if kernel == 'linear':
            K = linear(features)

        # All kernels in this control flow share the same structure
        elif kernel == 'rbf' or kernel == 'laplacian' or kernel == 'exponential':
            K = rbf(feature, features, sigma=self.sigma)

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

def rbf(feature, features, sigma=1.):
    """ Compute the rbf (AKA Gaussian) kernel.  """
    feature = feature.reshape(-1, 1)
    features = features.reshape(-1, 1)
    rbf = rbf_kernel(feature, Y=features, gamma=sigma)
    return rbf

def exponential(features, sigma=1.):
    """ Compute the exponential kernel"""
    pairwise_dists = squareform(pdist(features, 'euclidean'))
    exponential =  np.exp(-pairwise_dists / (2 * sigma ** 2))
    return exponential

def laplacian(features, sigma=1.):
    """ Compute the laplacian kernel"""
    pairwise_dists = squareform(pdist(features, 'euclidean'))
    laplacian =  np.exp(-pairwise_dists / sigma)
    return laplacian
