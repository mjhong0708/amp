#!/usr/bin/env python
"""
Folder that contains different regression methods.

"""

from ..utilities import ConvergenceOccurred

from scipy.optimize import fmin_bfgs




class Regressor:
    """Class to manage the regression of a generic model. That is, for a
    given parameter set, calculates the cost function (the difference in
    predicted energies and actual energies across training images), then
    decides how to adjust the parameters to reduce this cost function.
    Global optimization conditioners (e.g., simulated annealing, etc.) can
    be built into this class.
    """

    def __init__(self, optimizer=None, optimizer_kwargs=None):
        """optimizer can be specified; it should behave like a
        scipy.optimize optimizer. That is, it should take as its first two
        arguments the function to be optimized and the initial guess of the
        optimal paramters. Additional keyword arguments can be fed through
        the optimizer_kwargs dictionary."""
        #FIXME/ap optimizer could in principle be a list, if different
        # methods are to be used?
        if optimizer is None:
            from scipy.optimize import fmin_bfgs as optimizer
            optimizer_kwargs = {'gtol': 1e-15}
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs


    def regress(self, model, log):
        """Performs the regression. Calls model.get_cost_function,
        which should return the current value of the cost function
        until convergence has been reached, at which point it should
        raise a amp.utilities.ConvergenceException.
        """
        # FIXME/ap Optimizer also has space for fprime; needs
        # to be implemented. Especially important not to
        # call model functions twice to make this happen.
        log('Starting parameter optimization.', tic='opt')
        log(' Optimizer: %s' % self.optimizer)
        log(' Optimizer kwargs: %s' % self.optimizer_kwargs)
        x0 = model.get_vector()
        try:
            answer = self.optimizer(model.get_loss, x0,
                                    **self.optimizer_kwargs)
        except ConvergenceOccurred:
            log('...optimization successful.', toc='opt')
            return True
        else:
            log('...optimization unsuccessful.', toc='opt')
            return False


