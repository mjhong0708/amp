
from ..utilities import ConvergenceOccurred


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
            residual_per_atom = (predicted - actual) / len(image)
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
        print('%10.4e %10.4e %10.4e %10.4e' %
              (costfxn, self._energy_tol, max_residual, self._max_resid))
        converged = True
        if self._energy_tol is not None:
            if costfxn > self._energy_tol:
                converged = False
        if self._max_resid is not None:
            if max_residual > self._max_resid:
                converged = False
        return converged

