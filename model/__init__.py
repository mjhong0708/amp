
import numpy as np
from ase.calculators.calculator import Parameters
from ..utilities import (Logger, ConvergenceOccurred, make_sublists, now,
                         importer)
import warnings
try:
    from .. import fmodules
    fmodules_version = 5
    wrong_version = fmodules.check_version(version=fmodules_version)
    if wrong_version:
        raise RuntimeError('fortran modules are not updated. Recompile'
                           'with f2py as described in the README. '
                           'Correct version is %i.' % fmodules_version)
except ImportError:
    fmodules = None
    warnings.warn('Did not find fortran modules for use.')


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

    def get_energy(self, fingerprints):
        """Returns the model-predicted energy for an image, based on its
        fingerprint.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            energy = 0.0
            for index, (symbol, atomicfingerprint) in enumerate(fingerprints):
                atom_energy = self.get_atomic_energy(afp=atomicfingerprint,
                                                     index=index,
                                                     symbol=symbol)
                energy += atom_energy
        return energy

    def get_forces(self, fingerprints, fingerprintprimes):
        """Returns the model-predicted forces for an image, based on
        derivatives of fingerprints.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            forces = np.zeros((len(selfindices), 3))
            for (selfindex, selfsymbol, nindex, nsymbol, i), derafp in \
                    fingerprintprimes.iteritems():
                afp = fingerprints[nindex][1]
                dforce = self.get_force(afp=afp,
                                        derafp=derafp,
                                        nindex=nindex,
                                        nsymbol=nsymbol,
                                        direction=i,)
                forces[selfindex][i] += dforce
            return forces


class LossFunction:

    """Basic cost function, which can be used by the model.get_cost_function
    method which is required in standard model classes.
    This version is pure python and thus will be slow compared to a
    fortran/parallel implementation.

    If cores is None, it will pull it from the model itself. Only use
    this keyword to override the model's specification.

    Also has parallelization methods built in.

    See self.default_parameters for the default values of parameters
    specified as None.
    """

    default_parameters = {'convergence': {'energy_rmse': 0.001,
                                          'energy_maxresid': None,
                                          'force_rmse': 0.005,
                                          'force_maxresid': None, }
                          }

    def __init__(self, energy_coefficient=1.0, force_coefficient=0.04,
                 convergence=None, cores=None,
                 raise_ConvergenceOccurred=True,):
        p = self.parameters = Parameters(
            {'importname': '.model.LossFunction'})
        # 'dict' creates a copy; otherwise mutable in class.
        p['convergence'] = dict(self.default_parameters['convergence'])
        if convergence is not None:
            for key, value in convergence.iteritems():
                p['convergence'][key] = value
        p['energy_coefficient'] = energy_coefficient
        p['force_coefficient'] = force_coefficient
        self.raise_ConvergenceOccurred = raise_ConvergenceOccurred
        self._step = 0
        self._initialized = False
        self._cores = cores

    def attach_model(self, model, fingerprints=None,
                     fingerprintprimes=None, images=None):
        """Attach the model to be used to the loss function. fingerprints and
        training images need not be supplied if they are already attached to
        the model via model.trainingparameters."""
        self._model = model
        self.fingerprints = fingerprints
        self.fingerprintprimes = fingerprintprimes
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
        # FIXME: AKh: ap, should it decide whether or not to train forces based
        # on the value of force_coefficient?
        if ((self.parameters.force_coefficient != 0.) and
                (self.fingerprintprimes is None)):
            self.fingerprintprimes = \
                self._model.trainingparameters.descriptor.fingerprintprimes
        if self.images is None:
            self.images = self._model.trainingparameters.trainingimages

        if self._cores != 1:  # Initialize workers.
            import zmq
            from socket import gethostname
            from getpass import getuser
            pxssh = importer('pxssh')
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
        log('\n')
        if (convergence['force_rmse'] is None) and \
                (convergence['force_maxresid'] is None):
            log('\n  %12s  %12s' % ('EnergyLoss', 'MaxResid'))
            log('  %12s  %12s' % ('==========', '========'))
        else:
            log('%5s %19s %12s %12s %12s %12s' %
                ('', '', '', 'Energy',
                 '', 'Force'))
            log('%5s %19s %12s %12s %12s %12s' %
                ('Step', 'Time', 'EnergyLoss', 'MaxResid',
                 'ForceLoss', 'MaxResid'))
            log('%5s %19s %12s %12s %12s %12s' %
                ('='*5, '='*19, '='*12, '='*12,
                 '='*12, '='*12))

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

    def f(self, parametervector, complete_output=False):
        """Returns the current value of the loss function for a given set of
        parameters, or, if the energy is less than the energy_tol raises a
        ConvergenceException.

        By default only returns the loss function (needed for optimizers).
        Can return more information like max_residual if complete_output
        is True.
        """
        self._initialize()

        if self._cores == 1:
            self._model.vector = parametervector

            if self._model.fortran:
                num_images = len(self.images)
                energy_coefficient = self.parameters.energy_coefficient
                force_coefficient = self.parameters.force_coefficient
                if force_coefficient == 0.:
                    train_forces = False
                else:
                    train_forces = True
                mode = self._model.parameters.mode
                # FIXME: Should be corrected for image-centered:
                if mode == 'atom-centered':
                    num_atoms = None

                (actual_energies, actual_forces, elements, atomic_positions,
                 num_images_atoms, atomic_numbers, raveled_fingerprints,
                 num_neighbors, raveled_neighborlists,
                 raveled_fingerprintprimes) = (None,) * 10

                value = ravel_data(train_forces,
                                   mode,
                                   self.images,
                                   self.fingerprints,
                                   self.fingerprintprimes,)

                if mode == 'image-centered':
                    if not train_forces:
                        (actual_energies, atomic_positions) = value
                    else:
                        (actual_energies, actual_forces,
                         atomic_positions) = value
                else:
                    if not train_forces:
                        (actual_energies, elements, num_images_atoms,
                         atomic_numbers, raveled_fingerprints) = value
                    else:
                        (actual_energies, actual_forces, elements,
                         num_images_atoms, atomic_numbers,
                         raveled_fingerprints, num_neighbors,
                         raveled_neighborlists,
                         raveled_fingerprintprimes) = value

                send_data_to_fortran(fmodules,
                                     energy_coefficient,
                                     force_coefficient,
                                     train_forces,
                                     num_atoms,
                                     num_images,
                                     actual_energies,
                                     actual_forces,
                                     atomic_positions,
                                     num_images_atoms,
                                     atomic_numbers,
                                     raveled_fingerprints,
                                     num_neighbors,
                                     raveled_neighborlists,
                                     raveled_fingerprintprimes,
                                     self._model)

                (loss, dloss_dparameters, energy_loss, force_loss,
                 energy_maxresid, force_maxresid) = \
                    fmodules.calculate_f_and_fprime(
                    parameters=parametervector,
                    num_parameters=len(parametervector),
                    complete_output=True)

                fmodules.deallocate_variables()

            else:
                loss, dloss_dparameters, energy_loss, force_loss, \
                    energy_maxresid, force_maxresid = self.calculate_loss(
                        complete_output=True)
        else:
            server = self._sessions['master']
            processes = self._sessions['workers']

            # Subdivide tasks.
            keys = make_sublists(self.images.keys(), len(processes))

            args = {'task': 'f', 'complete_output': complete_output}

            results = self.process_parallels(parametervector,
                                             server,
                                             processes,
                                             keys,
                                             args=args)
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
            self.energy_maxresid, self.force_maxresid = \
            loss, dloss_dparameters, \
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
        """Returns the current value of the loss function for a given set of
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

                if self._model.fortran:
                    num_images = len(self.images)
                    mode = self._model.parameters.mode
                    energy_coefficient = self.parameters.energy_coefficient
                    force_coefficient = self.parameters.force_coefficient
                    if force_coefficient == 0.:
                        train_forces = False
                    else:
                        train_forces = True
                    # FIXME: Should be corrected for image-centered:
                    if mode == 'atom-centered':
                        num_atoms = None

                    (actual_energies, actual_forces, elements,
                     atomic_positions, num_images_atoms, atomic_numbers,
                     raveled_fingerprints, num_neighbors,
                     raveled_neighborlists,
                     raveled_fingerprintprimes) = (None,) * 10

                    value = ravel_data(train_forces,
                                       mode,
                                       self.images,
                                       self.fingerprints,
                                       self.fingerprintprimes,)

                    if mode == 'image-centered':
                        if not train_forces:
                            (actual_energies, atomic_positions) = value
                        else:
                            (actual_energies, actual_forces,
                             atomic_positions) = value
                    else:
                        if not train_forces:
                            (actual_energies, elements, num_images_atoms,
                             atomic_numbers,
                             raveled_fingerprints) = value
                        else:
                            (actual_energies, actual_forces, elements,
                             num_images_atoms, atomic_numbers,
                             raveled_fingerprints, num_neighbors,
                             raveled_neighborlists,
                             raveled_fingerprintprimes) = value

                    send_data_to_fortran(fmodules,
                                         energy_coefficient,
                                         force_coefficient,
                                         train_forces,
                                         num_atoms,
                                         num_images,
                                         actual_energies,
                                         actual_forces,
                                         atomic_positions,
                                         num_images_atoms,
                                         atomic_numbers,
                                         raveled_fingerprints,
                                         num_neighbors,
                                         raveled_neighborlists,
                                         raveled_fingerprintprimes,
                                         self._model)

                    (loss, dloss_dparameters, energy_loss, force_loss,
                     energy_maxresid, force_maxresid) = \
                        fmodules.calculate_f_and_fprime(
                        parameters=parametervector,
                        num_parameters=len(parametervector),
                        complete_output=True)

                    fmodules.deallocate_variables()

                else:
                    loss, dloss_dparameters, energy_loss, force_loss, \
                        energy_maxresid, force_maxresid = \
                        self.calculate_loss(complete_output=True)
            else:
                server = self._sessions['master']
                processes = self._sessions['workers']

                # Subdivide tasks.
                keys = make_sublists(self.images.keys(), len(processes))

                args = {'task': 'fprime', 'complete_output': complete_output}

                results = self.process_parallels(parametervector,
                                                 server,
                                                 processes,
                                                 keys,
                                                 args=args)
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

        if complete_output is False:
            return self.dloss_dparameters
        else:
            return {'loss': self.loss,
                    'dloss_dparameters': self.dloss_dparameters,
                    'energy_loss': self.energy_loss,
                    'force_loss': self.force_loss,
                    'energy_maxresid': self.energy_maxresid,
                    'force_maxresid': self.force_maxresid, }

    def calculate_loss(self, complete_output):
        """Method that calculates the loss, derivative of the loss with respect
        to parameters (if requested), and max_residual.
        """
        p = self.parameters
        energyloss = 0.
        forceloss = 0.
        energy_maxresid = 0.
        force_maxresid = 0.
        dloss_dparameters = None
        for hash, image in self.images.iteritems():
            no_of_atoms = len(image)
            amp_energy = self._model.get_energy(self.fingerprints[hash])
            actual_energy = image.get_potential_energy(apply_constraint=False)
            residual_per_atom = abs(amp_energy - actual_energy) / \
                len(image)
            if residual_per_atom > energy_maxresid:
                energy_maxresid = residual_per_atom
            energyloss += residual_per_atom**2

            # Calculates derivative of the loss function with respect to
            # parameters if complete_output is true
            if complete_output:
                if self._model.parameters.mode == 'image-centered':
                    raise NotImplementedError('This needs to be coded.')
                elif self._model.parameters.mode == 'atom-centered':
                    for atom in image:
                        symbol = atom.symbol
                        index = atom.index
                        afp = self.fingerprints[hash][index][1]
                        temp = self._model.get_dEnergy_dParameters(afp,
                                                                   index,
                                                                   symbol)
                        if dloss_dparameters is None:
                            dloss_dparameters = \
                                p.energy_coefficient * 2. * \
                                (amp_energy - actual_energy) * temp / \
                                (no_of_atoms ** 2.)
                        else:
                            dloss_dparameters += \
                                p.energy_coefficient * 2. * \
                                (amp_energy - actual_energy) * temp / \
                                (no_of_atoms ** 2.)

            if p.force_coefficient != 0.:
                amp_forces = \
                    self._model.get_forces(self.fingerprints[hash],
                                           self.fingerprintprimes[hash])
                actual_forces = image.get_forces(apply_constraint=False)
                for i in xrange(3):
                    for index in xrange(no_of_atoms):
                        force_resid = abs(amp_forces[index][i] -
                                          actual_forces[index][i])
                        if force_resid > force_maxresid:
                            force_maxresid = force_resid
                        forceloss += \
                            (1. / 3.) * (amp_forces[index][i] -
                                         actual_forces[index][i]) ** 2. / \
                            no_of_atoms
                # Calculates derivative of the loss function with respect to
                # parameters if complete_output is true
                if complete_output:
                    if self._model.parameters.mode == 'image-centered':
                        raise NotImplementedError('This needs to be coded.')
                    elif self._model.parameters.mode == 'atom-centered':
                        for key, derafp in \
                                self.fingerprintprimes[hash].iteritems():
                            (selfindex, selfsymbol, nindex, nsymbol, i) = key
                            afp = self.fingerprints[hash][nindex][1]
                            temp = \
                                self._model.get_dForce_dParameters(
                                    afp=afp,
                                    derafp=derafp,
                                    direction=i,
                                    nindex=nindex,
                                    nsymbol=nsymbol,)
                            dloss_dparameters += p.force_coefficient * \
                                (2.0 / 3.0) * \
                                (- amp_forces[selfindex][i] +
                                 actual_forces[selfindex][i]) * \
                                temp \
                                / no_of_atoms

        loss = p.energy_coefficient * energyloss + \
            p.force_coefficient * forceloss
        dloss_dparameters = np.array(dloss_dparameters)

        return loss, dloss_dparameters, energyloss, forceloss, \
            energy_maxresid, force_maxresid

    # All incoming requests will be dictionaries with three keys.
    # d['id']: process id number, assigned when process created above.
    # d['subject']: what the message is asking for / telling you.
    # d['data']: optional data passed from worker.

    def process_parallels(self, vector, server, processes, keys, args):
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
                elif request == 'fortran':
                    server.send_pyobj(self._model.fortran)
                elif request == 'modelstring':
                    server.send_pyobj(self._model.tostring())
                elif request == 'lossfunctionstring':
                    server.send_pyobj(self.parameters.tostring())
                elif request == 'fingerprints':
                    server.send_pyobj({k: self.fingerprints[k] for k in
                                       keys[int(message['id'])]})
                elif request == 'fingerprintprimes':
                    server.send_pyobj({k: self.fingerprintprimes[k] for k in
                                       keys[int(message['id'])]})
                elif request == 'args':
                    server.send_pyobj(args)
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
            energy_rmse = np.sqrt(energy_loss / len(self.images))
            if energy_rmse > p.convergence['energy_rmse']:
                energy_rmse_converged = False
        energy_maxresid_converged = True
        if p.convergence['energy_maxresid'] is not None:
            if energy_maxresid > p.convergence['energy_maxresid']:
                energy_maxresid_converged = False
        if self.parameters.force_coefficient != 0.:
            force_rmse_converged = True
            if p.convergence['force_rmse'] is not None:
                force_rmse = np.sqrt(force_loss / len(self.images))
                if force_rmse > p.convergence['force_rmse']:
                    force_rmse_converged = False
            force_maxresid_converged = True
            if p.convergence['force_maxresid'] is not None:
                if force_maxresid > p.convergence['force_maxresid']:
                    force_maxresid_converged = False

            log('%5i %19s %10.4e %1s %10.4e %1s %10.4e %1s %10.4e %1s' %
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
    for key, value in fprange.iteritems():
        fprange[key] = np.array(value)
    return fprange


def ravel_data(train_forces,
               mode,
               images,
               fingerprints,
               fingerprintprimes,):
    """
    Reshapes data of images into lists.
    """
    from ase.data import atomic_numbers as an

    actual_energies = [image.get_potential_energy(apply_constraint=False)
                       for hash, image in images.iteritems()]

    if mode == 'atom-centered':
        num_images_atoms = [len(image)
                            for hash, image in images.iteritems()]
        atomic_numbers = [an[atom.symbol]
                          for hash, image in images.iteritems()
                          for atom in image]

        def ravel_fingerprints(images,
                               fingerprints):
            """
            Reshape fingerprints of images into a list.
            """
            raveled_fingerprints = []
            elements = []
            for hash, image in images.iteritems():
                for index in range(len(image)):
                    elements += [fingerprints[hash][index][0]]
                    raveled_fingerprints += [fingerprints[hash][index][1]]
            elements = sorted(set(elements))
            # Could also work without images:
#            raveled_fingerprints = [afp
#                    for hash, value in fingerprints.iteritems()
#                    for (element, afp) in value]
            return elements, raveled_fingerprints

        elements, raveled_fingerprints = ravel_fingerprints(images,
                                                            fingerprints)
    else:
        atomic_positions = [image.positions.ravel()
                            for hash, image in images.iteritems()]

    if train_forces is True:

        actual_forces = \
            [image.get_forces(apply_constraint=False)[index]
             for hash, image in images.iteritems()
             for index in range(len(image))]

        if mode == 'atom-centered':

            def ravel_neighborlists_and_fingerprintprimes(images,
                                                          fingerprintprimes):
                """
                Reshape neighborlists and fingerprintprimes of images into a
                list and a matrix, respectively.
                """
                # Only neighboring atoms of type II (within the main cell)
                # need to be sent to fortran for force training.
                # All keys in fingerprintprimes are for type II neighborhoods.
                # Also note that each atom is considered as neighbor of
                # itself in fingerprintprimes.
                num_neighbors = []
                raveled_neighborlists = []
                raveled_fingerprintprimes = []
                for hash, image in images.iteritems():
                    for atom in image:
                        selfindex = atom.index
                        selfsymbol = atom.symbol
                        selfneighborindices = []
                        selfneighborsymbols = []
                        for key, derafp in fingerprintprimes[hash].iteritems():
                            # key = (selfindex, selfsymbol, nindex, nsymbol, i)
                            # i runs from 0 to 2. neighbor indices and symbols
                            # should be added just once.
                            if key[0] == selfindex and key[4] == 0:
                                selfneighborindices += [key[2]]
                                selfneighborsymbols += [key[3]]

                        neighborcount = 0
                        for nindex, nsymbol in zip(selfneighborindices,
                                                   selfneighborsymbols):
                            raveled_neighborlists += [nindex]
                            neighborcount += 1
                            for i in range(3):
                                fpprime = fingerprintprimes[hash][(selfindex,
                                                                   selfsymbol,
                                                                   nindex,
                                                                   nsymbol,
                                                                   i)]
                                raveled_fingerprintprimes += [fpprime]
                        num_neighbors += [neighborcount]

                return (num_neighbors,
                        raveled_neighborlists,
                        raveled_fingerprintprimes)

            (num_neighbors,
             raveled_neighborlists,
             raveled_fingerprintprimes) = \
                ravel_neighborlists_and_fingerprintprimes(images,
                                                          fingerprintprimes)
    if mode == 'image-centered':
        if not train_forces:
            return (actual_energies, atomic_positions)
        else:
            return (actual_energies, actual_forces, atomic_positions)
    else:
        if not train_forces:
            return (actual_energies, elements, num_images_atoms,
                    atomic_numbers, raveled_fingerprints)
        else:
            return (actual_energies, actual_forces, elements, num_images_atoms,
                    atomic_numbers, raveled_fingerprints, num_neighbors,
                    raveled_neighborlists, raveled_fingerprintprimes)


def send_data_to_fortran(_fmodules,
                         energy_coefficient,
                         force_coefficient,
                         train_forces,
                         num_atoms,
                         num_images,
                         actual_energies,
                         actual_forces,
                         atomic_positions,
                         num_images_atoms,
                         atomic_numbers,
                         raveled_fingerprints,
                         num_neighbors,
                         raveled_neighborlists,
                         raveled_fingerprintprimes,
                         model):
    """
    Function that sends images data to fortran code. Is used just once on each
    core.
    """
    from ase.data import atomic_numbers as an

    if model.parameters.mode == 'image-centered':
        mode_signal = 1
    elif model.parameters.mode == 'atom-centered':
        mode_signal = 2

    _fmodules.images_props.num_images = num_images
    _fmodules.images_props.actual_energies = actual_energies
    if train_forces:
        _fmodules.images_props.actual_forces = actual_forces

    _fmodules.model_props.energy_coefficient = energy_coefficient
    _fmodules.model_props.force_coefficient = force_coefficient
    _fmodules.model_props.train_forces = train_forces
    _fmodules.model_props.mode_signal = mode_signal

    if model.parameters.mode == 'atom-centered':
        fprange = model.parameters.fprange
        elements = sorted(fprange.keys())
        num_elements = len(elements)
        elements_numbers = [an[elm] for elm in elements]
        min_fingerprints = \
            [[fprange[elm][_][0] for _ in range(len(fprange[elm]))]
             for elm in elements]
        max_fingerprints = [[fprange[elm][_][1]
                             for _
                             in range(len(fprange[elm]))]
                            for elm in elements]
        num_fingerprints_of_elements = \
            [len(fprange[elm]) for elm in elements]

        _fmodules.images_props.num_elements = num_elements
        _fmodules.images_props.elements_numbers = elements_numbers
        _fmodules.images_props.num_images_atoms = num_images_atoms
        _fmodules.images_props.atomic_numbers = atomic_numbers
        if train_forces:
            _fmodules.images_props.num_neighbors = num_neighbors
            _fmodules.images_props.raveled_neighborlists = \
                raveled_neighborlists

        _fmodules.fingerprint_props.num_fingerprints_of_elements = \
            num_fingerprints_of_elements
        _fmodules.fingerprint_props.raveled_fingerprints = raveled_fingerprints
        _fmodules.neuralnetwork.min_fingerprints = min_fingerprints
        _fmodules.neuralnetwork.max_fingerprints = max_fingerprints
        if train_forces:
            _fmodules.fingerprint_props.raveled_fingerprintprimes = \
                raveled_fingerprintprimes
    else:
        _fmodules.images_props.num_atoms = num_atoms
        _fmodules.images_props.atomic_positions = atomic_positions

    # for neural neyworks only
    if model.parameters['importname'] == '.model.neuralnetwork.NeuralNetwork':

        hiddenlayers = model.parameters.hiddenlayers
        activation = model.parameters.activation

        if model.parameters.mode == 'atom-centered':
            from collections import OrderedDict
            no_layers_of_elements = \
                [3 if isinstance(hiddenlayers[elm], int)
                 else (len(hiddenlayers[elm]) + 2)
                 for elm in elements]
            nn_structure = OrderedDict()
            for elm in elements:
                len_of_fps = len(fprange[elm])
                if isinstance(hiddenlayers[elm], int):
                    nn_structure[elm] = \
                        ([len_of_fps] + [hiddenlayers[elm]] + [1])
                else:
                    nn_structure[elm] = \
                        ([len_of_fps] +
                         [layer for layer in hiddenlayers[elm]] + [1])

            no_nodes_of_elements = [nn_structure[elm][_]
                                    for elm in elements
                                    for _ in range(len(nn_structure[elm]))]

        else:
            num_atoms = model.parameters.num_atoms
            if isinstance(hiddenlayers, int):
                no_layers_of_elements = [3]
            else:
                no_layers_of_elements = [len(hiddenlayers) + 2]
            if isinstance(hiddenlayers, int):
                nn_structure = ([3 * num_atoms] + [hiddenlayers] + [1])
            else:
                nn_structure = ([3 * num_atoms] +
                                [layer for layer in hiddenlayers] + [1])
            no_nodes_of_elements = [nn_structure[_]
                                    for _ in range(len(nn_structure))]

        _fmodules.neuralnetwork.no_layers_of_elements = no_layers_of_elements
        _fmodules.neuralnetwork.no_nodes_of_elements = no_nodes_of_elements
        if activation == 'tanh':
            activation_signal = 1
        elif activation == 'sigmoid':
            activation_signal = 2
        elif activation == 'linear':
            activation_signal = 3
        _fmodules.neuralnetwork.activation_signal = activation_signal
