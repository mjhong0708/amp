#!/usr/bin/env python
# This module is the implementation of kernel ridge regression model into Amp.
# Author: Muammar El Khatib <muammarelkhatib@gmail.com>

from . import Model
from ase.calculators.calculator import Parameters
import numpy as np

from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.kernel_ridge import KernelRidge


class KRR(Model):
    """Class implementing Kernelized Ridge Regression

    Parameters
    ----------
    kernel : str
        Choose the kernel. Default is 'linear'.
    sigma : float
        Conditioning factor.
    """
    def __init__(self, sigma=1., kernel='linear', lamda=None, degree=3, coef0=1,
            kernel_parwms=None, mode=None, version=None, fortran=True):
        p = self.parameters = Parameters()
        p.version = version
        p.importname = '.model.kernel_ridge.KRR'
        p.mode = mode

        self.sigma = sigma
        self.kernel = kernel

    def kernel_matrix(self, afp, kernel='rbf'):
        """This method takes as arguments a feature vector and a string that refers
        to the kernel type used.

        Parameters
        ----------
        features : list or numpy array
            Column vector containing the fingerprint of the atoms in images.
        kernel : str
            Select the kernel to be used. Supported kernels are: 'linear',
            rbf'.

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

        afp = np.asarray(afp)
        print(afp)

        if kernel == 'linear':
            K = linear(afp)

        elif kernel == 'rbf':
            K = [ rbf(x, afp, sigma=self.sigma) for x in afp ]

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
        return K

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
        self._parallel = parallel
        self._log = log

        tp = self.trainingparameters = Parameters()
        tp.trainingimages = trainingimages
        tp.descriptor = descriptor
        tp.fingerprints = tp.descriptor.fingerprints

        feature_matrix = []
        suma = 0
        for hash, image in tp.trainingimages.iteritems():
            suma += 1
            features = []
            for element, afp in tp.fingerprints[hash]:
                #afp = np.asarray(afp)[:, np.newaxis]    # I append in column vectors
                print(element)
                afp = np.asarray(afp)
                features.append(afp)
                #features.append(self.kernel_matrix(afp, kernel=self.kernel)) I don't think this is correct

            feature_matrix.append(features)
            #kernel = self.kernel_matrix(features[0], kernel=self.kernel)
            print(suma)
            #print(kernel)
        for _ in feature_matrix:
            kernel = self.kernel_matrix(_, kernel=self.kernel)
            print(kernel)
            exit()

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
        """Force training"""
        pass

"""
Auxiliary functions to compute kernels
"""
def linear(x):
    """ Compute a linear kernel """
    linear = np.dot(afp, afp.T)
    return linear

def rbf(x, afp, sigma=1.):
    """ Compute the rbf (AKA Gaussian) kernel.  """
    rbf = np.exp(- (x - afp)**2 / (2 * sigma**2))
    return rbf
