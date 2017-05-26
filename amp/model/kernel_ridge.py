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
    alpha : float
        Conditioning factor.
    """
    def __init__(self, alpha=1, kernel='linear', gamma=None, degree=3, coef0=1,
            kernel_parwms=None, mode=None, version=None, fortran=True):
        p = self.parameters = Parameters()
        p.version = version
        p.importname = '.model.kernel_ridge.KRR'
        p.mode = mode

        self.alpha = alpha
        self.kernel=kernel

        self.clf = KernelRidge(alpha=self.alpha)

    def fit(self, trainingimages, descriptor, log, parallel):
        """Method to fit"""
        print(trainingimages)
        print(descriptor)
        self._parallel = parallel
        self._log = log

        tp = self.trainingparameters = Parameters()
        tp.trainingimages = trainingimages
        tp.descriptor = descriptor
        tp.fingerprints = tp.descriptor.fingerprints

        print(tp.fingerprints)

        images = []
        fingerprints = []
        hashes = []

        for hash, image in tp.trainingimages.iteritems():
            #print(image.get_potential_energy())
            hashes.append(hash)
            images.append(image.get_potential_energy())
            _fingerprints = []
            for element in tp.fingerprints[hash]:
                _fingerprints.append(element[1])
                #print(_fingerprints)
                #print(len(_fingerprints))
            fingerprints.append(_fingerprints)
            print(len(fingerprints))

        # Converting type from list to np.array
        fingerprints = np.array(fingerprints)
        targets = images
        print(targets)
        print(fingerprints.shape)
        n_targets = fingerprints.shape[0]
        n_features = fingerprints.shape[1] * fingerprints.shape[2]
        print(n_targets, n_features)
        features = fingerprints.reshape(n_targets, n_features)

        """
        print(features.shape)
        K = np.dot(features, features.T)
        print(K)
        I = np.identity(K.shape[0])
        print(K)
        print(I)
        sample_weight = np.empty(n_targets)
        copy = "precomputed"
        self.dual_coef_ = _solve_cholesky_kernel(K, targets, self.alpha, sample_weight, copy)
        print(self.dual_coef_)

        print(np.dot(features[0], self.dual_coef_))
        """
        self.clf.fit(features, targets)

        print(self.clf.predict(features[0]))

        exit()
        pass

    @property
    def forcetraining(self):
        """Force training"""
        pass

