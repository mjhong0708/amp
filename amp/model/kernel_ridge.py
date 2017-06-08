#!/usr/bin/env python
# This module is the implementation of kernel ridge regression model into Amp.
# Author: Muammar El Khatib <muammarelkhatib@gmail.com>

from . import Model
from ..regression import Regressor
from ase.calculators.calculator import Parameters
import numpy as np

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
    """
    def __init__(self, sigma=1., kernel='linear', lamda=1, degree=3,
            coef0=1, kernel_parwms=None, regressor=None, mode=None,
            version=None, fortran=True):
        p = self.parameters = Parameters()
        p.version = version
        p.importname = '.model.kernel_ridge.KRR'
        p.mode = mode
        self.regressor = regressor

        self.sigma = sigma
        self.kernel = kernel
        self.lamda = lamda

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

        tp = self.trainingparameters = Parameters()
        tp.trainingimages = trainingimages
        tp.descriptor = descriptor
        tp.fingerprints = tp.descriptor.fingerprints
        tp.fingerprintprimes = tp.descriptor.fingerprintprimes

        self.energies = []
        feature_matrix = []
        suma = 0
        for hash, image in tp.trainingimages.iteritems():
            energy = image.get_potential_energy()
            self.energies.append(energy)
            suma += 1
            features = []
            for element, afp in tp.fingerprints[hash]:
                #afp = np.asarray(afp)[:, np.newaxis]    # I append in column vectors
                print(element)
                print(afp)
                afp = np.asarray(afp)
                features.append(afp)
                #features.append(self.kernel_matrix(afp, kernel=self.kernel)) I don't think this is correct

            feature_matrix.append(features)
            #kernel = self.kernel_matrix(features[0], kernel=self.kernel)
            print(suma)
            #print(kernel)
        kij = []
        suma = 0
        for _ in feature_matrix:
            suma += 1
            print(np.asarray(_).shape)
            kernel = self.kernel_matrix(_, kernel=self.kernel)
            print()
            print(kernel)
            print(kernel.shape)
            from sklearn.metrics.pairwise import rbf_kernel
            kernel_sci = rbf_kernel(_, Y=None, gamma=1.)
            print(kernel_sci)
            if kernel.all() == kernel_sci.all():
                print('SCIKIT-LEARN AND AMP HAVE THE SAME KERNEL RBF MATRIX')
            print(suma)
            alphas = np.ones(np.asarray(kernel[0]).shape[-1])
            print(alphas)
            kij.append(kernel.dot(alphas))
            print(kij)

        self.kij = np.asarray(kij)
        print((np.asarray(self.kij).shape))

        print('alphas %s' % np.asarray(alphas).shape)
        resultado = fmin(self.get_loss, alphas, maxfun=99999999, maxiter=9999999999)
        print(resultado)

        print('Real energies: %s' % self.energies)
        for kernel_image in self.kij:
            print(kernel_image.dot(resultado))

        print(dir(tp.descriptor.fingerprints))
        exit()

    def get_loss(self, alphas):
        """Calculate the loss function with parameters alpha."""

        term2 = term1 = 0
        predictions = [ _.dot(alphas) for _ in self.kij ]
        print(predictions)
        print(self.energies)
        diffs = np.array(self.energies) - np.array(predictions)
        term1 = np.dot(diffs, diffs)
        for _k in self.kij:
            term2 += np.array(self.lamda) * alphas.dot(_k)

        self._losses.append([term1, term2])
        return float(term1 + term2)

        #"""
        #Create a kernel dict
        #"""
        #k = { }

        #for element, afp in tp.fingerprints.iteritems():
        #    print(element)
        #    print(afp)
        #    print('Call kernel')
        #    k[element] = self.kernel_matrix(afp, kernel=self.kernel)
        #print(k)

        #for hash, image in tp.trainingimages.iteritems():
        #    print(hash)
        #    #print(dir(image))

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
