
import numpy as np
from ase.calculators.calculator import Parameters
from ..utilities import Logger, ConvergenceOccurred, make_sublists, now


class Model(object):

    """
    Class that includes common methods between different models.
    """

    @property
    def log(self):
        """Method to set or get a logger. Should be an instance of
        amp.utilities.Logger."""
        if hasattr(self, '_log'):
            return self._log
        if hasattr(self.parent, 'log'):
            return self.parent.log
        return Logger(None)

    @log.setter
    def log(self, log):
        self._log = log

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to re-establish the calculator."""
        return self.parameters.tostring()

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

    def __init__(self, energy_coefficient, force_coefficient,
                 convergence, cores=None, raise_ConvergenceOccurred=True,):
        p = self.parameters = Parameters(
            {'importname': '.model.LossFunction'})
        p['convergence'] = convergence
        self.raise_ConvergenceOccurred = raise_ConvergenceOccurred
        self._step = 0
        self._initialized = False
        self._cores = cores
        self.energy_coefficient = energy_coefficient
        self.force_coefficient = force_coefficient

    def attach_model(self, model, fingerprints=None,
                     derfingerprints=None, images=None):
        """Attach the model to be used to the loss function. fingerprints and
        training images need not be supplied if they are already attached to
        the model via model.trainingparameters."""
        self._model = model
        self.fingerprints = fingerprints
        self.derfingerprints = derfingerprints
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
        if self.force_coefficient != 0.:  # ap: Is this a good place?
            self.derfingerprints = \
                self._model.trainingparameters.descriptor.derfingerprints
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
        convergence = p['convergence']
        log(' Loss function convergence criteria:')
        log('  energy_rmse: ' + str(convergence['energy_rmse']))
        log('  energy_maxresid: ' + str(convergence['energy_maxresid']))
        log('  force_rmse: ' + str(convergence['force_rmse']))
        log('  force_maxresid: ' + str(convergence['force_maxresid']))
        if (convergence['force_rmse'] is None) and \
                (convergence['force_maxresid'] is None):
            log('\n  %12s  %12s' % ('EnergyLoss', 'MaxResid'))
            log('  %12s  %12s' % ('==========', '========'))
        else:
            log('\n  %12s  %12s  %12s  %12s' % ('EnergyLoss', 'EnergyMaxResid',
                                                'ForceLoss', 'ForceMaxResid'))
            log('  %12s  %12s  %12s  %12s' % ('==========',
                                              '==========',
                                              '==========',
                                              '==========',))

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

    def f(self, parametervector, prime=True, complete_output=False):
        """Returns the current value of teh cost function for a given set of
        parameters, or, if the energy is less than the energy_tol raises a
        ConvergenceException.

        By default only returns the loss function (needed for optimizers).
        Can return more information like max_residual if complete_output
        is True.
        """
        self._initialize()

        if self._cores == 1:
            self._model.vector = parametervector
            loss, dloss_dparameters, energy_loss, force_loss, \
                energy_maxresid, force_maxresid = self.calculate_loss(prime)
        else:
            server = self._sessions['master']
            processes = self._sessions['workers']

            # Subdivide tasks.
            keys = make_sublists(self.images.keys(), len(processes))

            results = self.process_parallels(parametervector, server,
                                             processes, keys)
            loss = results['loss']
            dloss_dparameters = results['dloss_dparameters']
            energy_loss = results['energy_loss']
            force_loss = results['force_loss']
            energy_maxresid = results['energy_maxresid']
            force_maxresid = results['force_maxresid']

        if self.raise_ConvergenceOccurred:
            converged = self.check_convergence(energy_loss, force_loss,
                                               energy_maxresid, force_maxresid)
            if converged:
                self._model.vector = parametervector
                self._cleanup()
                raise ConvergenceOccurred()

        self.loss, self.dloss_dparameters, self.energy_loss, self.force_loss, \
        self.energy_maxresid, self.force_maxresid = loss, dloss_dparameters, \
        energy_loss, force_loss, energy_maxresid, force_maxresid

        if complete_output is False:
            return self.loss
        else:
            return {'loss': self.loss,
                    'dloss_dparameters': self.dloss_dparameters,
                    'energy_loss': self.energy_loss,
                    'force_loss': self.force_loss,
                    'energy_maxresid': self.energy_maxresid,
                    'force_maxresid': self.force_maxresid, }

    def fprime(self, parametervector, complete_output=False):
        """Returns the current value of teh cost function for a given set of
        parameters, or, if the energy is less than the energy_tol raises a
        ConvergenceException.

        By default only returns the loss function (needed for optimizers).
        Can return more information like max_residual if complete_output
        is True.
        """
        if self._step == 0:

            self._initialize()

            if self._cores == 1:
                self._model.vector = parametervector
                loss, dloss_dparameters, energy_loss, force_loss, \
                    energy_maxresid, force_maxresid = \
                    self.calculate_loss(prime=True)
            else:
                server = self._sessions['master']
                processes = self._sessions['workers']

                # Subdivide tasks.
                keys = make_sublists(self.images.keys(), len(processes))

                results = self.process_parallels(parametervector, server,
                                                 processes, keys)
                loss = results['loss']
                dloss_dparameters = results['dloss_dparameters']
                energy_loss = results['energy_loss']
                force_loss = results['force_loss']
                energy_maxresid = results['energy_maxresid']
                force_maxresid = results['force_maxresid']

            self.loss, self.dloss_dparameters, self.energy_loss, \
            self.force_loss, self.energy_maxresid, self.force_maxresid = \
                loss, dloss_dparameters, energy_loss, force_loss, \
                energy_maxresid, force_maxresid

        return self.dloss_dparameters

    def calculate_loss(self, prime):
        """Method that calculates the loss, derivative of the loss with respect
        to parameters (if requested), and max_residual.
        """
        energyloss = 0.
        forceloss = 0.
        energy_maxresid = 0.
        force_maxresid = 0.
        for hash, image in self.images.iteritems():
            no_of_atoms = len(image)
            predicted_energy = self._model.get_energy(self.fingerprints[hash])
            actual_energy = image.get_potential_energy(apply_constraint=False)
            residual_per_atom = abs(predicted_energy - actual_energy) / \
                len(image)
            if residual_per_atom > energy_maxresid:
                energy_maxresid = residual_per_atom
            energyloss += residual_per_atom**2

            # Calculates derivative of the loss function with respect to
            # parameters if prime is true
            if prime:
                if self._model.parameters.mode == 'image-centered':
                    raise NotImplementedError('This needs to be coded.')
                elif self._model.parameters.mode == 'atom-centered':
                    count = 0
                    for atom in image:
                        symbol = atom.symbol
                        index = atom.index
                        if count == 0:
                            dloss_dparameters = \
                                self.energy_coefficient * 2. * \
                                (predicted_energy - actual_energy) * \
                                self._model.get_dEnergy_dParameters(index,
                                                                    symbol) / \
                                (no_of_atoms ** 2.)
                        else:
                            dloss_dparameters += \
                                self.energy_coefficient * 2. * \
                                (predicted_energy - actual_energy) * \
                                self._model.get_dEnergy_dParameters(index,
                                                                    symbol) / \
                                (no_of_atoms ** 2.)
                        count += 1

            if self.force_coefficient != 0.:
                predicted_forces = \
                    self._model.get_forces(self.derfingerprints[hash])
                actual_forces = image.get_forces(apply_constraint=False)
                for i in xrange(3):
                    for index in xrange(no_of_atoms):
                        residual_force = abs(predicted_forces[index][i] -
                                             actual_forces[index][i])
                        if residual_force > force_maxresid:
                            force_maxresid = residual_force
                        forceloss += (1. / 3.) * (predicted_forces[index][i] -
                                                  actual_forces[index][i]) ** 2. / \
                                                  no_of_atoms
                # Calculates derivative of the loss function with respect to
                # parameters if prime is true
                if prime:
                    if self._model.parameters.mode == 'image-centered':
                        raise NotImplementedError('This needs to be coded.')
                    elif self._model.parameters.mode == 'atom-centered':
                        for key in self.derfingerprints[hash]:
                            (selfindex, selfsymbol, nindex, nsymbol, i) = key
                            # Reduce memory
                            self.derfingerprints[hash].pop(key)
                            temp = \
                                self._model.get_dForce_dParameters(i,
                                                                   nindex,
                                                                   nsymbol,)
                            dloss_dparameters += self.force_coefficient * \
                                (2.0 / 3.0) * \
                                (- predicted_forces[selfindex][i] +
                                 actual_forces[selfindex][i]) * \
                                temp \
                                / no_of_atoms
        energyloss = energyloss / len(self.images)
        forceloss = forceloss / len(self.images)
        loss = self.energy_coefficient * energyloss + \
            self.force_coefficient * forceloss
        dloss_dparameters = dloss_dparameters / len(self.images)
        dloss_dparameters = np.array(dloss_dparameters)

        return loss, dloss_dparameters, energyloss, forceloss, \
            energy_maxresid, force_maxresid

    # All incoming requests will be dictionaries with three keys.
    # d['id']: process id number, assigned when process created above.
    # d['subject']: what the message is asking for / telling you.
    # d['data']: optional data passed from worker.

    def process_parallels(self, vector, server, processes, keys):
        # For each process
        finished = np.array([False] * len(processes))
        results = {'loss': 0.,
                   'dloss_dparameters': [0.] * len(vector),
                   'energy_loss': 0.,
                   'force_loss': 0.,
                   'energy_maxresid': 0.,
                   'force_maxresid': 0.}
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
                elif request == 'derfingerprints':
                    server.send_pyobj({k: self.derfingerprints[k] for k in
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
                results['dloss_dparameters'] += result['dloss_dparameters']
                results['energy_loss'] += result['energy_loss']
                results['force_loss'] += result['force_loss']
                if result['energy_maxresid'] > results['energy_maxresid']:
                    results['energy_maxresid'] = result['energy_maxresid']
                if result['force_maxresid'] > results['force_maxresid']:
                    results['force_maxresid'] = result['force_maxresid']
                finished[int(message['id'])] = True
        return results

    def check_convergence(self, energy_loss, force_loss,
                          energy_maxresid, force_maxresid):
        """Checks to see whether convergence is met; if it is, raises
        ConvergenceException to stop the optimizer."""
        p = self.parameters
        energy_rmse_converged = True
        log = self._model.log
        if p.convergence['energy_rmse'] is not None:
            energy_rmse = np.sqrt(energy_loss)
            if energy_rmse > p.convergence['energy_rmse']:
                energy_rmse_converged = False
        energy_maxresid_converged = True
        if p.convergence['energy_maxresid'] is not None:
            if energy_maxresid > p.convergence['energy_maxresid']:
                energy_maxresid_converged = False
        if self.force_coefficient != 0.:
            force_rmse_converged = True
            if p.convergence['force_rmse'] is not None:
                force_rmse = np.sqrt(force_loss)
                if force_rmse > p.convergence['force_rmse']:
                    force_rmse_converged = False
            force_maxresid_converged = True
            if p.convergence['force_maxresid'] is not None:
                if force_maxresid > p.convergence['force_maxresid']:
                    force_maxresid_converged = False

            log(' %5i  %19s %12.4e %1s %12.4e %1s %12.4e %1s %12.4e %1s' %
                (self._step, now(), energy_rmse,
                 'C' if energy_rmse_converged else '',
                 energy_maxresid,
                 'C' if energy_maxresid_converged else '',
                 force_rmse,
                 'C' if force_rmse_converged else '',
                 force_maxresid,
                 'C' if force_maxresid_converged else ''))
            self._step += 1
            return energy_rmse_converged and energy_maxresid_converged and \
                force_rmse_converged and force_maxresid_converged
        else:
            log(' %5i  %19s %12.4e %1s %12.4e %1s' %
                (self._step, now(), energy_rmse,
                 'C' if energy_rmse_converged else '',
                 energy_maxresid,
                 'C' if energy_maxresid_converged else ''))
            self._step += 1
            return energy_rmse_converged and energy_maxresid_converged


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
