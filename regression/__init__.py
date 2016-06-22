from ..utilities import ConvergenceOccurred


class Regressor:

    """Class to manage the regression of a generic model. That is, for a
    given parameter set, calculates the cost function (the difference in
    predicted energies and actual energies across training images), then
    decides how to adjust the parameters to reduce this cost function.
    Global optimization conditioners (e.g., simulated annealing, etc.) can
    be built into this class.

    :param lossprime: Decides whether or not the regressor needs to be fed in
                      by gradient of the loss function as well as the loss
                      function itself.
    :type lossprime: boolean

    """

    def __init__(self, optimizer=None, optimizer_kwargs=None, lossprime=True):
        """optimizer can be specified; it should behave like a
        scipy.optimize optimizer. That is, it should take as its first two
        arguments the function to be optimized and the initial guess of the
        optimal paramters. Additional keyword arguments can be fed through
        the optimizer_kwargs dictionary."""
        if optimizer is None:
            from scipy.optimize import fmin_bfgs as optimizer
            optimizer_kwargs = {'gtol': 1e-500}
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lossprime = lossprime

    def regress(self, model, log):
        """Performs the regression. Calls model.get_loss,
        which should return the current value of the loss function
        until convergence has been reached, at which point it should
        raise a amp.utilities.ConvergenceException.
        """
        # FIXME/ap Optimizer also has space for fprime; needs
        # to be implemented. Especially important not to
        # call model functions twice to make this happen.
        log('Starting parameter optimization.', tic='opt')
        log(' Optimizer: %s' % self.optimizer)
        log(' Optimizer kwargs: %s' % self.optimizer_kwargs)
        x0 = model.vector.copy()
        try:
            if self.lossprime:
                self.optimizer(model.get_loss, x0, model.get_lossprime,
                               **self.optimizer_kwargs)
            else:
                self.optimizer(model.get_loss, x0, **self.optimizer_kwargs)
        except ConvergenceOccurred:
            log('...optimization successful.', toc='opt')
            return True
        else:
            log('...optimization unsuccessful.', toc='opt')
            max_lossprime = max(abs(max(model.lossfunction.dloss_dparameters)),
                                abs(min(model.lossfunction.dloss_dparameters)))
            log('...maximum absolute value of loss prime: %.3e'
                % max_lossprime)
            return False
