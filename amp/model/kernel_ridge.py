#!/usr/bin/env python
# This module is the implementation of kernel ridge regression model into Amp.
# Author: Muammar El Khatib <muammarelkhatib@gmail.com>

import os
from . import LossFunction, Model
from ..regression import Regressor
from ase.calculators.calculator import Parameters
import numpy as np
from ..utilities import ConvergenceOccurred, make_filename

from scipy.optimize import fmin
from scipy.spatial.distance import pdist, squareform

from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.kernel_ridge import KernelRidge


class KRR(Model):
    """Class implementing Kernelized Ridge Regression

    Parameters
    ----------
    kernel : str
        Choose the kernel. Default is 'linear'.
    sigma : float
        Length scale of the Gaussian in the case of RBF. Default is 1.
    lamda : float
        Strength of the regularization.
    checkpoints : int
        Frequency with which to save parameter checkpoints upon training. E.g.,
        100 saves a checpoint on each 100th training setp.  Specify None for no
        checkpoints.
    lossfunction : object
        Loss function object, if at all desired by the user.
    """
    def __init__(self, sigma=1., kernel='linear', lamda=1, degree=3, coef0=1,
            kernel_parwms=None, weights=None, regressor=None, mode=None,
            version=None, fortran=False, checkpoints=100, lossfunction=None):
        p = self.parameters = Parameters()
        p.version = version
        p.importname = '.model.kernel_ridge.KRR'
        p.mode = mode
        p.weights = weights

        self.regressor = regressor
        self.parent = None  # Can hold a reference to main Amp instance.
        self.sigma = sigma
        self.fortran = fortran
        self.kernel = kernel
        self.lamda = lamda
        self.checkpoints = checkpoints
        self.lossfunction = lossfunction

        if self.lossfunction is None:
            self.lossfunction = LossFunction()

        self._losses = []

    def kernel_matrix(self, features, kernel='rbf'):
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
        print(features)

        if kernel == 'linear':
            K = linear(features)

        # All kernels in this control flow share the same structure
        elif kernel == 'rbf' or kernel == 'laplacian' or kernel == 'exponential':
            K = rbf(features, sigma=self.sigma)

            """This is for testing purposes
            xjs = np.array([
                0.,
                0.05263158,
                0.10526316,
                0.15789474,
                0.21052632,
                0.26315789,
                0.31578947,
                0.36842105,
                0.42105263,
                0.47368421,
                0.52631579,
                0.57894737,
                0.63157895,
                0.68421053,
                0.73684211,
                0.78947368,
                0.84210526,
                0.89473684,
                0.94736842,
                1.
                ])
            K = np.array([rbf(x, xjs, sigma=self.sigma) for x in xjs])
            print(K)
            """
        else:
            raise NotImplementedError('This kernel needs to be coded.')

        return np.asarray(K)

    def fit(self, trainingimages, descriptor, log, parallel):
        """This method fits the kernel ridge model using a loss function.

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

        #print(trainingimages)
        #print(descriptor)
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

        """
        This checks that we need fingerprintprime.
        """
        if self.forcetraining == True:
            tp.fingerprintprimes = tp.descriptor.fingerprintprimes

        self.energies = []
        self.feature_matrix = []
        self.hashes = []
        suma = 0
        for hash, image in tp.trainingimages.iteritems():
            energy = image.get_potential_energy()
            self.energies.append(energy)
            suma += 1
            features = []
            self.hashes.append(hash)
            for element, afp in tp.fingerprints[hash]:
                #afp = np.asarray(afp)[:, np.newaxis]    # I append in column vectors
                print(element)
                print(afp)
                afp = np.asarray(afp)
                features.append(afp)
                #features.append(self.kernel_matrix(afp, kernel=self.kernel)) I don't think this is correct

            self.feature_matrix.append(features)
            #kernel = self.kernel_matrix(features[0], kernel=self.kernel)
            print(suma)
            #print(kernel)


        if p.mode is None:
            p.mode = descriptor.parameters.mode


        kij = []
        self.kernel_dict = {}
        suma = 0
        for index, _ in enumerate(self.feature_matrix):
            suma += 1
            print(np.asarray(_).shape)
            kernel = self.kernel_matrix(_, kernel=self.kernel)
            print()
            print(kernel)
            from sklearn.metrics.pairwise import rbf_kernel
            kernel_sci = rbf_kernel(_, Y=None, gamma=1.)
            print(kernel_sci)
            if kernel.all() == kernel_sci.all():
                print('SCIKIT-LEARN AND AMP HAVE THE SAME KERNEL RBF MATRIX')
            print(suma)
            alphas = np.ones(np.asarray(kernel[0]).shape[-1])
            print(kernel)
            print(np.sum(kernel, axis=0))
            self.kernel_dict[self.hashes[index]] = np.sum(kernel, axis=0)
            print(kernel.dot(alphas))
            print(p.weights)
            kij.append(np.sum(kernel, axis=0))

        self.kij = np.asarray(kij)
        print((np.asarray(self.kij).shape))
        print(self.kernel_dict)
        print('Kernel dict')

        if p.weights is None:
            log('Initializing weights.')
            if p.mode == 'image-centered':
                raise NotImplementedError('Needs to be coded.')
            elif p.mode == 'atom-centered':
                p.weights = np.ones(np.asarray(self.kij[0]).shape[-1])
                print(p.weights)

        #print('weights %s' % np.asarray(p.weights).shape)
        resultado = fmin(self._get_loss, p.weights, maxfun=99999999, maxiter=9999999999)
        print(resultado)

        print('Real energies: %s' % self.energies)
        for kernel_image in self.kij:
            print(kernel_image.dot(resultado))

        print(dir(tp.descriptor.fingerprints))

        self.step = 0
        result = self.regressor.regress(model=self, log=log)
        return result  # True / False

    @property
    def forcetraining(self):
        """Returns true if forcetraining is turned on (as determined by
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
        """Access to get or set the model parameters (weights, scaling for
        each network) as a single vector, useful in particular for
        regression.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        if self.parameters['weights'] is None:
            return None
        p = self.parameters
        #print('vector')
        #p = self.parameters
        #if (len(set(len(x) for x in self.kij)) <= 1) == True:
        #    vector = np.ones(np.asarray(self.kij[0]).shape[-1])
        vector = p.weights
        print(vector)
        return vector

    @vector.setter
    def vector(self, vector):
        p = self.parameters
        p['weights'] = p.weights

    def get_loss(self, vector):
        """Method to be called by the regression master.

        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the loss (cost) function.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        print('SOY LLAMADO')
        print(vector)
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
        loss = self.lossfunction.get_loss(np.asarray(vector), lossprime=False)['loss']
        print(loss)
        if hasattr(self, 'observer'):
            self.observer(self, vector, loss)
        self.step += 1
        return loss

    def _get_loss(self, weights):
        """Calculate the loss function with parameters alpha."""
        term2 = term1 = 0

        predictions = [ _.dot(weights) for _ in self.kij ]
        print(predictions)
        print(self.energies)
        diffs = np.array(self.energies) - np.array(predictions)
        term1 = np.dot(diffs, diffs)
        for _k in self.kij:
            term2 += np.array(self.lamda) * weights.dot(_k)
        self._losses.append([term1, term2])

        return float(term1 + term2)

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
        pass



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

    def calculate_atomic_energy(self, afp, index, symbol, hash=None):
        """
        Given input to the neural network, output (which corresponds to energy)
        is calculated about the specified atom. The sum of these for all
        atoms is the total energy (in atom-centered mode).

        Parameters
        ---------
        afp : list
            Atomic fingerprints in the form of a list to be used as input to
            the neural network.
        index: int
            Index of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).
        symbol : str
            Symbol of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).

        Returns
        -------
        atomic_amp_energy : float
            Energy.
        """
        if self.parameters.mode != 'atom-centered':
            raise AssertionError('calculate_atomic_energy should only be '
                                 ' called in atom-centered mode.')

        weight = self.parameters.weights[index]
        atomic_amp_energy = self.kernel_dict[hash][index] * weight
        print(atomic_amp_energy)
        return atomic_amp_energy
"""
Auxiliary functions to compute kernels
"""
def linear(features):
    """ Compute a linear kernel """
    linear = np.dot(features, features.T)
    return linear

def rbf(features, sigma=1.):
    """ Compute the rbf (AKA Gaussian) kernel.  """
    pairwise_dists = squareform(pdist(features, 'euclidean'))
    rbf =  np.exp(-pairwise_dists ** 2 / (sigma ** 2))
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
