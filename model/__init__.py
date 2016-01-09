
import numpy as np

from ..utilities import ConvergenceOccurred, Logger


class LossFunction:
    """Basic cost function, which can be used by the model.get_cost_function
    method which is required in standard model classes.
    This version is pure python and thus will be slow compared to a fortran/parallel implementation.
    
    """

    def __init__(self, energy_tol=0.001, max_resid=0.001):
        self._energy_tol = energy_tol
        self._max_resid = max_resid

    def attach_model(self, model):
        self._model = model

        self.log = Logger(None)
        if hasattr(model, 'trainingparameters'):
            if 'log' in model.trainingparameters:
                self.log = model.trainingparameters['log']
        self.log(' Loss function attached to model. Convergence criteria:')
        self.log('  energy_tol: ' + str(self._energy_tol))
        self.log('  max_resid: ' + str(self._max_resid))
        self.log('  %12s  %12s' % ('EnergyLoss', 'MaxResid'))
        self.log('  %12s  %12s' % ('==========', '========'))

    def __call__(self, parametervector):
        """Returns the current value of teh cost function for a given set of parameters,
        or, if the energy is less than the energy_tol raises a ConvergenceException.
        """
        #FIXME/ap If parallel, or fortran, implement that here.
        # There will need to be program instances (e.g., fortran
        # executables) that have the fingerprints, etc., already
        # stored and just calculate the cost function with the new
        # parameter vector.

        tp = self._model.trainingparameters
        self._model.set_vector(parametervector)
        costfxn = 0.
        max_residual = 0.
        for hash, image in tp.trainingimages.iteritems():
            predicted = self._model.get_energy(tp.fingerprint.fingerprints[hash])
            actual = image.get_potential_energy(apply_constraint=False)
            #print(len(image), predicted, actual)
            residual_per_atom = abs(predicted - actual) / len(image)
            if residual_per_atom > max_residual:
                max_residual = residual_per_atom
            costfxn += residual_per_atom**2
        costfxn = costfxn / len(tp.trainingimages)
        converged = self.check_convergence(costfxn, max_residual)

        if converged:
            # Make sure first step is done in case of switching to fortran.
            self._model.set_vector(parametervector)
            raise ConvergenceOccurred()
        return costfxn

    def check_convergence(self, costfxn, max_residual):
        """Checks to see whether convergence is met; if it is, raises
        ConvergenceException to stop the optimizer."""
        energyconverged = True
        maxresidconverged = True
        if self._energy_tol is not None:
            if costfxn > self._energy_tol:
                energyconverged = False
        if self._max_resid is not None:
            if max_residual > self._max_resid:
                maxresidconverged = False
        self.log('  %12.4e %1s %12.4e %1s' %
                 (costfxn, 'C' if energyconverged else '',
                  max_residual, 'C' if maxresidconverged else ''))
        return energyconverged and maxresidconverged


def calculate_fingerprints_range(fp, images):
    """Calculates the range for the fingerprints corresponding to images,
    stored in fp. fp is a fingerprints object with the fingerprints data
    stored in a dictionary-like object at fp.fingerprints. (Typically this
    is a .utilties.Data structure.) images is a hashed dictionary of atoms
    for which to consider the range.

    In image-centered mode, returns an array of (min, max) values for each
    fingerprint. In atom-centered mode, returns a dictionary of such
    arrays, one per element.
    """
    if fp.parameters.mode == 'image-centered':
        raise NotImplementedError()
    elif fp.parameters.mode == 'atom-centered':
        fprange = {}
        for hash, image in images.iteritems():
            imagefingerprints = fp.fingerprints[hash]
            for element, fingerprint in imagefingerprints:
                if element not in fprange:
                    fprange[element] = [[val, val] for val in
                                            fingerprint]
                else:
                    assert len(fprange[element]) == len(fingerprint)
                    for i, ridge in enumerate(fingerprint):
                        if ridge < fprange[element][i][0]:
                            fprange[element][i][0] = ridge
                        elif ridge > fprange[element][i][1]:
                            fprange[element][i][1] = ridge
    for key, value in fprange.items():
        fprange[key] = np.array(value)
    return fprange
