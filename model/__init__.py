
import numpy as np
from ase.calculators.calculator import Parameters

from ..utilities import ConvergenceOccurred, make_sublists, now


class Model(object):

    """
    Class that includes common methods between different models.
    """

    def get_energy(self, fingerprint):
        """Returns the model-predicted energy for an image, based on its
        fingerprint.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            energy = 0.0
            for index, (element, atomicfingerprint) in enumerate(fingerprint):
                atom_energy = self.get_atomic_energy(afp=atomicfingerprint,
                                                     index=index,
                                                     symbol=element)
                energy += atom_energy
        return energy

    def get_forces(self, derfingerprints):
        """Returns the model-predicted forces for an image, based on
        derivatives of fingerprints.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in derfingerprints.keys()])
            forces = np.zeros((len(selfindices), 3))
            while len(derfingerprints) > 0:
                (selfindex, selfsymbol, nindex, nsymbol, i), derafp = \
                    derfingerprints.popitem()  # Reduce memory.
                forces[selfindex][i] += self.get_atomic_force(direction=i,
                                                              derafp=derafp,
                                                              index=nindex,
                                                              symbol=nsymbol,)
            return forces


class LossFunction:

    """Basic cost function, which can be used by the model.get_cost_function
    method which is required in standard model classes.
    This version is pure python and thus will be slow compared to a
    fortran/parallel implementation.

    If cores is None, it will pull it from the model itself. Only use
    this keyword to override the model's specification.

    Also has parallelization methods built in.
    """

    def __init__(self, energy_tol=0.001, max_resid=0.001, cores=None,
                 raise_ConvergenceOccurred=True):
        p = self.parameters = Parameters(
            {'importname': '.model.LossFunction'})
        p['energy_tol'] = energy_tol
        p['max_resid'] = max_resid
        self.raise_ConvergenceOccurred = raise_ConvergenceOccurred
        self._step = 0
        self._initialized = False
        self._cores = cores

    def attach_model(self, model, fingerprints=None, images=None):
        """Attach the model to be used to the loss function. fingerprints and
        training images need not be supplied if they are already attached to
        the model via model.trainingparameters."""
        self._model = model
        self.fingerprints = fingerprints
        self.images = images

    def _initialize(self):
        """Procedures to be run on the first call only, such as establishing
        SSH sessions, etc."""
        if self._initialized is True:
            return

        if self._cores is None:
            self._cores = self._model.cores
        log = self._model.log

        if self.fingerprints is None:
            self.fingerprints = \
                self._model.trainingparameters.descriptor.fingerprints
        if self.images is None:
            self.images = self._model.trainingparameters.trainingimages

        if self._cores != 1:  # Initialize workers.
            import zmq
            import pxssh
            from socket import gethostname
            from getpass import getuser
            log(' Parallel processing.')
            context = zmq.Context()
            server = context.socket(zmq.REP)
            serverport = server.bind_to_random_port('tcp://*')
            serversocket = '%s:%s' % (gethostname(), serverport)
            log('  Established server at %s' % serversocket)

            module = self.__module__
            workercommand = ('python -m %s %%s %s' %
                             (module, serversocket))

            def establish_ssh(process_id):
                """Uses pxssh to establish a SSH connections and get the
                command running. process_id is an assigned unique identifier
                for each process. When the process starts, it needs to send
                <amp-connect>, followed by the location of its standard error,
                followed by <stderr>. Then it should communicate via zmq.
                """
                ssh = pxssh.pxssh()
                ssh.login(workerhostname, getuser())
                ssh.sendline(workercommand % process_id)
                ssh.expect('<amp-connect>')
                ssh.expect('<stderr>')
                log('  Session %i (%s): %s' %
                    (process_id, workerhostname, ssh.before.strip()))
                return ssh

            # Create processes over SSH.
            log('  Establishing workers:')
            processes = []
            for workerhostname, nprocesses in self._cores.iteritems():
                n = len(processes)
                processes.extend([establish_ssh(_ + n) for _ in
                                  range(nprocesses)])

            self._sessions = {'master': server,
                              'workers': processes}

        p = self.parameters
        log(' Loss function convergence criteria:')
        log('  energy_tol: ' + str(p['energy_tol']))
        log('  max_resid: ' + str(p['max_resid']))
        log('\n  %12s  %12s' % ('EnergyLoss', 'MaxResid'))
        log('  %12s  %12s' % ('==========', '========'))

        self._initialized = True

    def _cleanup(self):
        """Closes SSH sessions."""
        if not hasattr(self, '_sessions'):
            return
        server = self._sessions['master']

        def process_parallels():
            finished = np.array([False] * len(self._sessions['workers']))
            while not finished.all():
                message = server.recv_pyobj()
                if (message['subject'] == '<request>' and
                        message['data'] == 'parameters'):
                    server.send_pyobj('<stop>')
                    finished[int(message['id'])] = True

        process_parallels()
        for _ in self._sessions['workers']:
            _.logout()
        del self._sessions['workers']

    def __call__(self, parametervector, complete_output=False):
        """Returns the current value of teh cost function for a given set of
        parameters, or, if the energy is less than the energy_tol raises a
        ConvergenceException.

        By default only returns the cost function (needed for optimizers).
        Can return more information like max_residual if complete_output
        is True.
        """
        self._initialize()

        if self._cores == 1:
            self._model.vector = parametervector
            loss = 0.
            max_residual = 0.
            for hash, image in self.images.iteritems():
                predicted = self._model.get_energy(self.fingerprints[hash])
                actual = image.get_potential_energy(apply_constraint=False)
                residual_per_atom = abs(predicted - actual) / len(image)
                if residual_per_atom > max_residual:
                    max_residual = residual_per_atom
                loss += residual_per_atom**2
            loss = loss / len(self.images)
        else:
            server = self._sessions['master']
            processes = self._sessions['workers']

            # Subdivide tasks.
            keys = make_sublists(self.images.keys(), len(processes))

            # All incoming requests will be dictionaries with three keys.
            # d['id']: process id number, assigned when process created above.
            # d['subject']: what the message is asking for / telling you.
            # d['data']: optional data passed from worker.

            def process_parallels(vector):
                # For each process
                finished = np.array([False] * len(processes))
                results = {'loss': 0., 'max_residual': 0.}
                while not finished.all():
                    message = server.recv_pyobj()
                    if message['subject'] == '<purpose>':
                        server.send_string('calculate_loss_function')
                    elif message['subject'] == '<request>':
                        request = message['data']  # Variable name.
                        if request == 'images':
                            subimages = {k: self.images[k] for k in
                                         keys[int(message['id'])]}
                            server.send_pyobj(subimages)
                        elif request == 'modelstring':
                            server.send_pyobj(self._model.tostring())
                        elif request == 'lossfunctionstring':
                            server.send_pyobj(self.parameters.tostring())
                        elif request == 'fingerprints':
                            server.send_pyobj({k: self.fingerprints[k] for k in
                                               keys[int(message['id'])]})
                        elif request == 'parameters':
                            if finished[int(message['id'])]:
                                server.send_pyobj('<continue>')
                            else:
                                server.send_pyobj(vector)
                        else:
                            raise NotImplementedError()
                    elif message['subject'] == '<result>':
                        result = message['data']
                        server.send_string('meaningless reply')
                        results['loss'] += result['loss']
                        if result['max_residual'] > results['max_residual']:
                            results['max_residual'] = result['max_residual']
                        finished[int(message['id'])] = True
                return results

            results = process_parallels(parametervector)
            loss = results['loss']
            max_residual = results['max_residual']

        if self.raise_ConvergenceOccurred:
            converged = self.check_convergence(loss, max_residual)
            if converged:
                self._model.vector = parametervector
                self._cleanup()
                raise ConvergenceOccurred()

        if complete_output is False:
            return loss
        else:
            return {'loss': loss,
                    'max_residual': max_residual, }

    def check_convergence(self, loss, max_residual):
        """Checks to see whether convergence is met; if it is, raises
        ConvergenceException to stop the optimizer."""
        p = self.parameters
        energyconverged = True
        maxresidconverged = True
        log = self._model.log
        if p.energy_tol is not None:
            if loss > p.energy_tol:
                energyconverged = False
        if p.max_resid is not None:
            if max_residual > p.max_resid:
                maxresidconverged = False
        log(' %5i  %19s %12.4e %1s %12.4e %1s' %
            (self._step, now(), loss, 'C' if energyconverged else '',
             max_residual, 'C' if maxresidconverged else ''))
        self._step += 1
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
                    fprange[element] = [[_, _] for _ in fingerprint]
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
