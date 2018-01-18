#!/usr/bin/env python
# This module is the implementation of kernel ridge regression model into Amp.
#
# Author: Muammar El Khatib <muammarelkhatib@brown.edu>

import itertools
import threading
import time
import sys
import os
import numpy as np
from collections import OrderedDict
from scipy.linalg import cholesky

from ase.calculators.calculator import Parameters

from ..regression import Regressor
from ..utilities import (make_filename, hash_images, Logger,
                         ConvergenceOccurred, make_sublists, now,
                         setup_parallel)

try:
    from .. import fmodules
except ImportError:
    fmodules = None


class Model(object):
    """Class that includes common methods between different models."""

    @property
    def log(self):
        """Method to set or get a logger. Should be an instance of
        amp.utilities.Logger.

        Parameters
        ----------
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        """
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
        # Make sure numpy prints out enough data.
        np.set_printoptions(precision=30, threshold=999999999)
        return self.parameters.tostring()

    def calculate_energy(self, fingerprints, hash=None, trainingimages=None,
                         fp_trainingimages=None):
        """Calculates the model-predicted energy for an image, based on its
        fingerprint.

        Parameters
        ----------
        fingerprints : dict or list
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            self.atomic_energies = []
            energy = 0.0

            if not isinstance(fingerprints, list):
                fingerprints = fingerprints[hash]


            if self.cholesky is False:
                for index, (symbol, afp) in enumerate(fingerprints):
                    arguments = dict(
                            afp=afp,
                            index=index,
                            symbol=symbol,
                            )

                    if hash is not None:
                        arguments['hash'] = hash
                        arguments['fp_trainingimages'] = fp_trainingimages
                        arguments['kernel'] = self.parameters.kernel
                        arguments['sigma'] = self.parameters.sigma
                        arguments['trainingimages'] = trainingimages

                    atom_energy = self.calculate_atomic_energy(**arguments)
                    self.atomic_energies.append(atom_energy)
                    energy += atom_energy
            else:
                energy = self.total_energy_from_cholesky(
                        hash=hash,
                        fp_trainingimages=fp_trainingimages,
                        kernel=self.parameters.kernel,
                        trainingimages=trainingimages,
                        sigma=self.parameters.sigma,
                        fingerprints=fingerprints
                        )
        return energy

    def calculate_forces(self, fingerprints, fingerprintprimes, hash=None,
                         trainingimages=None, fp_trainingimages=None,
                         t_descriptor=None):
        """Calculates the model-predicted forces for an image, based on
        derivatives of fingerprints.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprint derivatives as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            forces = np.zeros((len(selfindices), 3))

            for selfindex, (symbol, afp) in enumerate(fingerprints):
                for component in range(3):
                    arguments = dict(
                            index=selfindex,
                            symbol=symbol,
                            component=component,
                            hash=hash,
                            t_descriptor=t_descriptor,
                            sigma=self.parameters.sigma,
                            trainingimages=trainingimages,
                            fingerprintprimes=fingerprintprimes
                            )
                    if self.cholesky is False:
                        dforce = self.calculate_force(**arguments)
                    else:
                        dforce = self.forces_from_cholesky(**arguments)

                    forces[selfindex][component] += dforce
        return forces

    def calculate_dEnergy_dParameters(self, fingerprints):
        """Calculates a list of floats corresponding to the derivative of
        model-predicted energy of an image with respect to model parameters.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            denergy_dparameters = None
            for index, (symbol, afp) in enumerate(fingerprints):
                temp = self.calculate_dAtomicEnergy_dParameters(afp=afp,
                                                                index=index,
                                                                symbol=symbol)
                if denergy_dparameters is None:
                    denergy_dparameters = temp
                else:
                    denergy_dparameters += temp
        return denergy_dparameters

    def calculate_numerical_dEnergy_dParameters(self, fingerprints, d=0.00001):
        """Evaluates dEnergy_dParameters using finite difference.

        This will trigger two calls to calculate_energy(), with each parameter
        perturbed plus/minus d.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        d : float
            The amount of perturbation in each parameter.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            vector = self.vector
            denergy_dparameters = []
            for _ in range(len(vector)):
                vector[_] += d
                self.vector = vector
                eplus = self.calculate_energy(fingerprints)
                vector[_] -= 2 * d
                self.vector = vector
                eminus = self.calculate_energy(fingerprints)
                denergy_dparameters += [(eplus - eminus) / (2 * d)]
                vector[_] += d
                self.vector = vector
            denergy_dparameters = np.array(denergy_dparameters)
        return denergy_dparameters

    def calculate_dForces_dParameters(self, fingerprints, fingerprintprimes):
        """Calculates an array of floats corresponding to the derivative of
        model-predicted atomic forces of an image with respect to model
        parameters.

        Parameters
        ----------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprint derivatives as values.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            dforces_dparameters = {(selfindex, i): None
                                   for selfindex in selfindices
                                   for i in range(3)}
            for key in fingerprintprimes.keys():
                selfindex, selfsymbol, nindex, nsymbol, i = key
                derafp = fingerprintprimes[key]
                afp = fingerprints[nindex][1]
                temp = self.calculate_dForce_dParameters(afp=afp,
                                                         derafp=derafp,
                                                         direction=i,
                                                         nindex=nindex,
                                                         nsymbol=nsymbol,)
                if dforces_dparameters[(selfindex, i)] is None:
                    dforces_dparameters[(selfindex, i)] = temp
                else:
                    dforces_dparameters[(selfindex, i)] += temp
        return dforces_dparameters

    def calculate_numerical_dForces_dParameters(self, fingerprints,
                                                fingerprintprimes, d=0.00001):
        """Evaluates dForces_dParameters using finite difference. This will
        trigger two calls to calculate_forces(), with each parameter perturbed
        plus/minus d.

        Parameters
        ---------
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprint derivatives as values.
        d : float
            The amount of perturbation in each parameter.
        """

        if self.parameters.mode == 'image-centered':
            raise NotImplementedError('This needs to be coded.')
        elif self.parameters.mode == 'atom-centered':
            selfindices = set([key[0] for key in fingerprintprimes.keys()])
            dforces_dparameters = {(selfindex, i): []
                                   for selfindex in selfindices
                                   for i in range(3)}
            vector = self.vector
            for _ in range(len(vector)):
                vector[_] += d
                self.vector = vector
                fplus = self.calculate_forces(fingerprints, fingerprintprimes)
                vector[_] -= 2 * d
                self.vector = vector
                fminus = self.calculate_forces(fingerprints, fingerprintprimes)
                for selfindex in selfindices:
                    for i in range(3):
                        dforces_dparameters[(selfindex, i)] += \
                            [(fplus[selfindex][i] - fminus[selfindex][i]) / (
                                2 * d)]
                vector[_] += d
                self.vector = vector
            for selfindex in selfindices:
                for i in range(3):
                    dforces_dparameters[(selfindex, i)] = \
                        np.array(dforces_dparameters[(selfindex, i)])
        return dforces_dparameters


class LossFunction:
    """Basic loss function, which can be used by the model.get_loss
    method which is required in standard model classes.

    This version is pure python and thus will be slow compared to a
    fortran/parallel implementation.

    If parallel is None, it will pull it from the model itself. Only use
    this keyword to override the model's specification.

    Also has parallelization methods built in.

    See self.default_parameters for the default values of parameters
    specified as None.

    Parameters
    ----------
    energy_coefficient : float
        Coefficient of the energy contribution in the loss function.
    force_coefficient : float
        Coefficient of the force contribution in the loss function.
        Can set to None as shortcut to turn off force training.
    convergence : dict
        Dictionary of keys and values defining convergence.  Keys are
        'energy_rmse', 'energy_maxresid', 'force_rmse', and 'force_maxresid'.
        If 'force_rmse' and 'force_maxresid' are both set to None, force
        training is turned off and force_coefficient is set to None.
    parallel : dict
        Parallel configuration dictionary. Will pull from model itself if
        not specified.
    overfit : float
        Multiplier of the weights norm penalty term in the loss function.
    raise_ConvergenceOccurred : bool
        If True will raise convergence notice.
    log_losses : bool
        If True will log the loss function value in the log file else will not.
    d : None or float
        If d is None, both loss function and its gradient are calculated
        analytically. If d is a float, then gradient of the loss function is
        calculated by perturbing each parameter plus/minus d.
    """

    default_parameters = {'convergence': {'energy_rmse': 0.001,
                                          'energy_maxresid': None,
                                          'force_rmse': None,
                                          'force_maxresid': None, }
                          }

    def __init__(self, energy_coefficient=1.0, force_coefficient=0.04,
                 convergence=None, parallel=None, overfit=0.,
                 raise_ConvergenceOccurred=True, log_losses=True, d=None):
        p = self.parameters = Parameters(
            {'importname': '.model.LossFunction'})
        # 'dict' creates a copy; otherwise mutable in class.
        c = p['convergence'] = dict(self.default_parameters['convergence'])
        if convergence is not None:
            for key, value in convergence.items():
                p['convergence'][key] = value
        p['energy_coefficient'] = energy_coefficient
        p['force_coefficient'] = force_coefficient
        p['overfit'] = overfit
        self.raise_ConvergenceOccurred = raise_ConvergenceOccurred
        self.log_losses = log_losses
        self.d = d
        self._step = 0
        self._initialized = False
        self._data_sent = False
        self._parallel = parallel
        if (c['force_rmse'] is None) and (c['force_maxresid'] is None):
            p['force_coefficient'] = None
        if p['force_coefficient'] is None:
            c['force_rmse'] = None
            c['force_maxresid'] = None

    def attach_model(self, model, fingerprints=None,
                     fingerprintprimes=None, images=None):
        """Attach the model to be used to the loss function.

        fingerprints and training images need not be supplied if they are
        already attached to the model via model.trainingparameters.

        Parameters
        ----------
        model : object
            Class representing the regression model.
        fingerprints : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprints as values.
        fingerprintprimes : dict
            Dictionary with images hashs as keys and the corresponding
            fingerprint derivatives as values.
        images : list or str
            List of ASE atoms objects with positions, symbols, energies, and
            forces in ASE format. This is the training set of data. This can
            also be the path to an ASE trajectory (.traj) or database (.db)
            file. Energies can be obtained from any reference, e.g. DFT
            calculations.
        """
        self._model = model
        self.fingerprints = fingerprints
        self.fingerprintprimes = fingerprintprimes
        self.images = images

    def _initialize(self, args):
        """Procedures to be run on the first call only, such as establishing
        SSH sessions, etc."""
        if self._initialized is True:
            return

        if self._parallel is None:
            self._parallel = self._model._parallel
        log = self._model.log

        if self.fingerprints is None:
            self.fingerprints = \
                self._model.trainingparameters.descriptor.fingerprints

        # May also make sense to decide whether or not to calculate
        # fingerprintprimes based on the value of train_forces.
        if ((self.parameters.force_coefficient is not None) and
                (self.fingerprintprimes is None)):
            self.fingerprintprimes = \
                self._model.trainingparameters.descriptor.fingerprintprimes
        if self.images is None:
            self.images = self._model.trainingparameters.trainingimages

        if self._parallel['cores'] != 1:
            # Initialize workers and send them parameters.

            python = sys.executable
            workercommand = '%s -m %s' % (python, self.__module__)
            self._sessions = setup_parallel(self._parallel, workercommand, log,
                                            setup_publisher=True)
            n_pids = self._sessions['n_pids']
            workerkeys = make_sublists(self.images.keys(), n_pids)
            server = self._sessions['master']
            setup_complete = np.array([False] * n_pids)
            while not setup_complete.all():
                message = server.recv_pyobj()
                if message['subject'] == 'purpose':
                    server.send_string('calculate_loss_function')
                elif message['subject'] == 'setup complete':
                    server.send_pyobj('thank you')
                    setup_complete[int(message['id'])] = True
                elif message['subject'] == 'request':
                    request = message['data']  # Variable name.
                    if request == 'images':
                        subimages = {k: self.images[k] for k in
                                     workerkeys[int(message['id'])]}
                        server.send_pyobj(subimages)
                    elif request == 'fortran':
                        server.send_pyobj(self._model.fortran)
                    elif request == 'modelstring':
                        server.send_pyobj(self._model.tostring())
                    elif request == 'lossfunctionstring':
                        server.send_pyobj(self.parameters.tostring())
                    elif request == 'fingerprints':
                        server.send_pyobj({k: self.fingerprints[k] for k in
                                           workerkeys[int(message['id'])]})
                    elif request == 'fingerprintprimes':
                        if self.fingerprintprimes is not None:
                            server.send_pyobj({k: self.fingerprintprimes[k]
                                               for k in
                                               workerkeys[int(message['id'])]})
                        else:
                            server.send_pyobj(None)
                    elif request == 'args':
                        server.send_pyobj(args)
                    elif request == 'publisher':
                        server.send_pyobj(self._sessions['publisher_socket'])
                    else:
                        raise NotImplementedError('Unknown request: {}'
                                                  .format(request))
            subscribers_working = np.array([False] * n_pids)

            def thread_function():
                """Broadcast from the background."""
                thread = threading.current_thread()
                while True:
                    if thread.abort is True:
                        break
                    self._sessions['publisher'].send_pyobj('test message')
                    time.sleep(0.1)

            thread = threading.Thread(target=thread_function)
            thread.abort = False  # to cleanly exit the thread
            thread.start()
            while not subscribers_working.all():
                message = server.recv_pyobj()
                server.send_pyobj('meaningless reply')
                if message['subject'] == 'subscriber working':
                    subscribers_working[int(message['id'])] = True
            thread.abort = True
            self._sessions['publisher'].send_pyobj('done')

        if self.log_losses:
            p = self.parameters
            convergence = p['convergence']
            log(' Loss function convergence criteria:')
            log('  energy_rmse: ' + str(convergence['energy_rmse']))
            log('  energy_maxresid: ' + str(convergence['energy_maxresid']))
            log('  force_rmse: ' + str(convergence['force_rmse']))
            log('  force_maxresid: ' + str(convergence['force_maxresid']))
            log(' Loss function set-up:')
            log('  energy_coefficient: ' + str(p.energy_coefficient))
            log('  force_coefficient: ' + str(p.force_coefficient))
            log('  overfit: ' + str(p.overfit))
            log('\n')
            if p.force_coefficient is None:
                header = '%5s %19s %12s %12s %12s'
                log(header %
                    ('', '', '', '', 'Energy'))
                log(header %
                    ('Step', 'Time', 'Loss (SSD)', 'EnergyRMSE', 'MaxResid'))
                log(header %
                    ('=' * 5, '=' * 19, '=' * 12, '=' * 12, '=' * 12))
            else:
                header = '%5s %19s %12s %12s %12s %12s %12s'
                log(header %
                    ('', '', '', '', 'Energy',
                     '', 'Force'))
                log(header %
                    ('Step', 'Time', 'Loss (SSD)', 'EnergyRMSE', 'MaxResid',
                     'ForceRMSE', 'MaxResid'))
                log(header %
                    ('=' * 5, '=' * 19, '=' * 12, '=' * 12, '=' * 12,
                     '=' * 12, '=' * 12))

        self._initialized = True

    def _send_data_to_fortran(self,):
        """Procedures to be run in fortran mode for a single requested core
        only. Also just on the first call for sending data to fortran modules.
        """
        if self._data_sent is True:
            return

        num_images = len(self.images)
        p = self.parameters
        energy_coefficient = p.energy_coefficient
        overfit = p.overfit
        if p.force_coefficient is None:
            train_forces = False
            force_coefficient = 0.
        else:
            train_forces = True
            force_coefficient = p.force_coefficient
        mode = self._model.parameters.mode
        if mode == 'atom-centered':
            num_atoms = None
        elif mode == 'image-centered':
            raise NotImplementedError('Image-centered mode is not coded yet.')

        (actual_energies, actual_forces, elements, atomic_positions,
         num_images_atoms, atomic_numbers, raveled_fingerprints, num_neighbors,
         raveled_neighborlists, raveled_fingerprintprimes) = (None,) * 10

        value = ravel_data(train_forces,
                           mode,
                           self.images,
                           self.fingerprints,
                           self.fingerprintprimes,)

        if mode == 'image-centered':
            if not train_forces:
                (actual_energies, atomic_positions) = value
            else:
                (actual_energies, actual_forces, atomic_positions) = value
        else:
            if not train_forces:
                (actual_energies, elements, num_images_atoms,
                 atomic_numbers, raveled_fingerprints) = value
            else:
                (actual_energies, actual_forces, elements, num_images_atoms,
                 atomic_numbers, raveled_fingerprints, num_neighbors,
                 raveled_neighborlists, raveled_fingerprintprimes) = value

        send_data_to_fortran(fmodules,
                             energy_coefficient,
                             force_coefficient,
                             overfit,
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
                             self._model,
                             self.d)
        self._data_sent = True

    def _cleanup(self):
        """Closes SSH sessions."""
        self._initialized = False
        if not hasattr(self, '_sessions'):
            return
        server = self._sessions['master']
        # Need to properly close socket connection (python3).
        server.close()

        for _ in self._sessions['connections']:
            if hasattr(_, 'logout'):
                _.logout()
        del self._sessions['connections']

    def get_loss(self, parametervector, energy_vector, energy_kernel,
                 forces_vector, forces_kernel, lossprime):
        """Returns the current value of the loss function for a given set of
        parameters, or, if the energy is less than the energy_tol raises a
        ConvergenceException.

        Parameters
        ----------
        parametervector : list
            Parameters of the regression model in the form of a list.
        lossprime : bool
            If True, will calculate and return dloss_dparameters, else will
            only return zero for dloss_dparameters.
        """

        self._initialize(args={'lossprime': lossprime, 'd': self.d})

        if self._parallel['cores'] == 1:
            if self._model.fortran:
                self._model.vector = parametervector
                self._send_data_to_fortran()
                (loss, dloss_dparameters, energy_loss, force_loss,
                 energy_maxresid, force_maxresid) = \
                    fmodules.calculate_loss(parameters=parametervector,
                                            num_parameters=len(
                                                parametervector),
                                            lossprime=lossprime)
            else:
                loss, dloss_dparameters, energy_loss, force_loss, \
                    energy_maxresid, force_maxresid = \
                    self.calculate_loss(parametervector,
                                        energy_vector,
                                        energy_kernel,
                                        forces_vector,
                                        forces_kernel,
                                        lossprime=lossprime)
        else:
            server = self._sessions['master']
            n_pids = self._sessions['n_pids']

            results = self.process_parallels(parametervector,
                                             server,
                                             n_pids)
            loss = results['loss']
            dloss_dparameters = results['dloss_dparameters']
            energy_loss = results['energy_loss']
            force_loss = results['force_loss']
            energy_maxresid = results['energy_maxresid']
            force_maxresid = results['force_maxresid']

        self.loss, self.energy_loss, self.force_loss, \
            self.energy_maxresid, self.force_maxresid = \
            loss, energy_loss, force_loss, energy_maxresid, force_maxresid

        if lossprime:
            self.dloss_dparameters = dloss_dparameters

        if self.raise_ConvergenceOccurred:
            # Only during calculation of loss function (and not lossprime)
            # convergence is checked and values are printed out in the log
            # file.
            if lossprime is False:
                self._model.vector = parametervector
                converged = self.check_convergence(loss,
                                                   energy_loss,
                                                   force_loss,
                                                   energy_maxresid,
                                                   force_maxresid)
                if converged:
                    self._cleanup()
                    raise ConvergenceOccurred()

        return {'loss': self.loss,
                'dloss_dparameters': (self.dloss_dparameters
                                      if lossprime is True
                                      else dloss_dparameters),
                'energy_loss': self.energy_loss,
                'force_loss': self.force_loss,
                'energy_maxresid': self.energy_maxresid,
                'force_maxresid': self.force_maxresid, }

    def calculate_loss(self, parametervector, energy_vector, energy_kernel,
                       forces_vector, forces_kernel, lossprime):
        """Method that calculates the loss, derivative of the loss with respect
        to parameters (if requested), and max_residual.

        Parameters
        ----------
        parametervector : list
            Parameters of the regression model in the form of a list.

        lossprime : bool
            If True, will calculate and return dloss_dparameters, else will
            only return zero for dloss_dparameters.
        """
        self._model.vector = parametervector
        p = self.parameters
        energyloss = 0.
        forceloss = 0.
        energy_maxresid = 0.
        force_maxresid = 0.
        dloss_dparameters = np.array([0.] * len(parametervector))
        model = self._model

        force_resid = 0.

        for hash in self.images.keys():
            image = self.images[hash]
            no_of_atoms = len(image)
            amp_energy = model.calculate_energy(
                    self.fingerprints[hash],
                    hash)
            actual_energy = image.get_potential_energy(
                    apply_constraint=False)
            residual_per_atom = abs(
                    amp_energy - actual_energy
                    ) / no_of_atoms

            if residual_per_atom > energy_maxresid:
                energy_maxresid = residual_per_atom
            energyloss += residual_per_atom ** 2

            if p.force_coefficient is not None:
                descriptor = self._model.trainingparameters.descriptor
                if model.numeric_force is False:
                    amp_forces = \
                        model.calculate_forces(
                                self.fingerprints[hash],
                                self.fingerprintprimes[hash],
                                hash=hash,
                                t_descriptor=descriptor
                                )

                actual_forces = image.get_forces(apply_constraint=False)
                for index in range(no_of_atoms):
                    temp_f = np.linalg.norm(
                            amp_forces[index] - actual_forces[index],
                            ord=1
                            )
                    force_resid += temp_f

                force_resid = force_resid / no_of_atoms

                if force_resid > force_maxresid:
                    force_maxresid = force_resid

                forceloss += (1. / 3.) * force_resid ** 2

        loss = energyloss * p.energy_coefficient

        if p.force_coefficient is not None:
            loss += p.force_coefficient * forceloss

        # if model.lamda coefficient is more than zero, overfit
        # contribution to loss and dloss_dparameters is also added.

        if model.lamda > 0.:
            overfitloss = 0.
            for symbol in energy_vector.keys():
                _vector = energy_vector[symbol]
                # Based on https://stats.stackexchange.com/a/70127/160746
                overfitloss += _vector.T.dot(energy_kernel.dot(_vector))
                if p.force_coefficient is not None:
                    for component in range(3):
                        _vector = forces_vector[symbol][component]
                        overfitloss += _vector.T.dot(forces_kernel[component].dot(_vector))
            overfitloss *= model.lamda
            loss += overfitloss

        return loss, dloss_dparameters, energyloss, forceloss, \
            energy_maxresid, force_maxresid

    # All incoming requests will be dictionaries with three keys.
    # d['id']: process id number, assigned when process created above.
    # d['subject']: what the message is asking for / telling you.
    # d['data']: optional data passed from worker.

    def process_parallels(self, vector, server, n_pids):
        """

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        server : object
            Master session of parallel processing.
        processes: list of objects
            Worker sessions for parallel processing.
        """
        # FIXME/ap: We don't need to pass in most of the arguments.
        # They are stored already.
        results = {'loss': 0.,
                   'dloss_dparameters': [0.] * len(vector),
                   'energy_loss': 0.,
                   'force_loss': 0.,
                   'energy_maxresid': 0.,
                   'force_maxresid': 0.}

        publisher = self._sessions['publisher']

        # Broadcast parameters for this call.
        publisher.send_pyobj(vector)

        # Receive the result.
        finished = np.array([False] * self._sessions['n_pids'])
        while not finished.all():
            message = server.recv_pyobj()
            server.send_pyobj('thank you')

            assert message['subject'] == 'result'
            result = message['data']

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

    def check_convergence(self, loss, energy_loss, force_loss,
                          energy_maxresid, force_maxresid):
        """Check convergence

        Checks to see whether convergence is met; if it is, raises
        ConvergenceException to stop the optimizer.

        Parameters
        ----------
        loss : float
            Value of the loss function.
        energy_loss : float
            Value of the energy contribution of the loss function.
        force_loss : float
            Value of the force contribution of the loss function.
        energy_maxresid : float
            Maximum energy residual.
        force_maxresid : float
            Maximum force residual.
        """
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
        if p.force_coefficient is not None:
            force_rmse_converged = True
            if p.convergence['force_rmse'] is not None:
                force_rmse = np.sqrt(force_loss / len(self.images))
                if force_rmse > p.convergence['force_rmse']:
                    force_rmse_converged = False
            force_maxresid_converged = True
            if p.convergence['force_maxresid'] is not None:
                if force_maxresid > p.convergence['force_maxresid']:
                    force_maxresid_converged = False

            if self.log_losses:
                log('%5i %19s %12.4e %10.4e %1s'
                    ' %10.4e %1s %10.4e %1s %10.4e %1s' %
                    (self._step, now(), loss, energy_rmse,
                     'C' if energy_rmse_converged else '-',
                     energy_maxresid,
                     'C' if energy_maxresid_converged else '-',
                     force_rmse,
                     'C' if force_rmse_converged else '-',
                     force_maxresid,
                     'C' if force_maxresid_converged else '-'))

            self._step += 1
            return energy_rmse_converged and energy_maxresid_converged and \
                force_rmse_converged and force_maxresid_converged
        else:
            if self.log_losses:
                log('%5i %19s %12.4e %10.4e %1s %10.4e %1s' %
                    (self._step, now(), loss, energy_rmse,
                     'C' if energy_rmse_converged else '-',
                     energy_maxresid,
                     'C' if energy_maxresid_converged else '-'))
            self._step += 1
            return energy_rmse_converged and energy_maxresid_converged


class KRR(Model):
    """Class implementing Kernelized Ridge Regression in Amp

    Parameters
    ----------
    sigma : float
        Length scale of the Gaussian in the case of RBF, exponential, and
        laplacian kernels. Default is 1.
    kernel : str
        Choose the kernel. Available kernels are: 'linear', 'rbf', 'laplacian',
        and 'exponential'. Default is 'rbf'.
    lamda : float
        Strength of the regularization in the loss function when minimizing
        error.
    weights : dict
        Dictionary of weights.
    regressor : object
        Regressor class to be used.
    mode : str
        Atom- or image-centered mode.
    trainingimages : str
        Path to Trajectory file containing the images in the training set. This
        is useful for predicting new structures.
    version : str
        Version.
    fortran : bool
        Use fortran code.
    checkpoints : int
        Frequency with which to save parameter checkpoints upon training. E.g.,
        100 saves a checkpoint on each 100th training setp.  Specify None for
        no checkpoints. Default is None.
    lossfunction : object
        Loss function object.
    cholesky : bool
        Wether or not we are using Cholesky decomposition to determine the
        weights.
    weights_independent : bool
        Wheter or not the weights are going to be split for energy and forces.
    numeric_force : bool
        Use numeric_force of atom energy predicted by Amp to minimize the loss
        function. This is not yet implemented.
    """
    def __init__(self, sigma=1., kernel='rbf', lamda=0., weights=None,
                 regressor=None, mode=None, trainingimages=None, version=None,
                 fortran=False, checkpoints=None, lossfunction=None,
                 cholesky=False, weights_independent=True,
                 numeric_force=False):

        np.set_printoptions(precision=30, threshold=999999999)

        # Version check, particularly if restarting.
        compatibleversions = ['2015.12', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use KRR'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               'newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        p = self.parameters = Parameters()
        p.importname = '.model.kernel_ridge.KRR'
        p.version = version
        p.weights = weights
        p.weights_independent = self.weights_independent = weights_independent
        p.mode = mode
        p.kernel = self.kernel = kernel
        p.sigma = self.sigma = sigma
        p.lamda = self.lamda = lamda
        p.cholesky = self.cholesky = cholesky
        p.numeric_force = self.numeric_force = numeric_force
        p.trainingimages = self.trainingimages = trainingimages

        self.regressor = regressor
        self.parent = None  # Can hold a reference to main Amp instance.
        self.fortran = fortran
        self.checkpoints = checkpoints
        self.lossfunction = lossfunction
        self.properties = []

        self.kernel_e = {}  # Kernel dictionary for energies
        self.kernel_f = {}  # Kernel dictionary for forces

        if self.lossfunction is None:
            self.lossfunction = LossFunction()

    def fit(self, trainingimages, descriptor, log, parallel, only_setup=False):
        """Fit kernel ridge model

        This function is capable to fit KRR using either a L2 loss function or
        matrix factorization in the case when the cholesky keyword argument is
        set to True.

        Parameters
        ----------
        trainingimages : dict
            Hashed dictionary of training images.
        descriptor : object
            Class with local chemical environments of atoms.
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        parallel: dict
            Parallel configuration dictionary. Takes the same form as in
            amp.Amp.
        """

        # Set all parameters and report to logfile.
        self._parallel = parallel
        self._log = log

        if self.regressor is None and self.cholesky is False:
            from ..regression import Regressor
            self.regressor = Regressor()

        p = self.parameters
        tp = self.trainingparameters = Parameters()
        tp.trainingimages = trainingimages
        tp.descriptor = descriptor
        tp.fingerprints = tp.descriptor.fingerprints

        if p.mode is None:
            p.mode = descriptor.parameters.mode
        else:
            assert p.mode == descriptor.parameters.mode
        log('Regression in %s mode.' % p.mode)

        if len(list(self.kernel_e.keys())) == 0:

            log('Calculating %s kernel...' % self.kernel, tic='kernel')
            log('Parameters:')
            log(' lamda = %s' % self.lamda)
            log(' sigma = %s' % self.sigma)
            kij_args = dict(
                    trainingimages=tp.trainingimages,
                    fp_trainingimages=tp.fingerprints,
                    )

            if self.fortran is True:
                kij_args['only_features'] = True

            self.get_energy_kernel(**kij_args)
            self.properties.append('energy')

            if self.forcetraining is True:
                self.properties.append('forces')
                kijf_args = dict(
                    trainingimages=tp.trainingimages,
                    t_descriptor=tp.descriptor
                    )
                if self.fortran is True:
                    kijf_args['only_features'] = True
                self.get_forces_kernel(**kijf_args)

            log('...kernel computed in', toc='kernel')

        if p.weights is None:
            log('Initializing weights.')
            if p.mode == 'image-centered':
                raise NotImplementedError('Needs to be coded.')
            elif p.mode == 'atom-centered':
                self.size = len(self.reference_features)
                weights = OrderedDict()
                for prop in self.properties:
                    weights[prop] = OrderedDict()
                    for hash in tp.trainingimages.keys():
                        imagefingerprints = tp.fingerprints[hash]
                        for element, fingerprint in imagefingerprints:
                            if (element not in weights and prop is 'energy'):
                                weights[prop][element] = np.random.uniform(
                                        low=-1.0,
                                        high=1.0,
                                        size=(self.size))
                            elif (element not in weights and prop is 'forces'):
                                if p.weights_independent is True:
                                    weights[prop][element] = np.random.uniform(
                                            low=-1.0,
                                            high=1.0,
                                            size=(3, self.size)
                                            )
                                else:
                                    weights[prop][element] = np.ones(self.size)
                p.weights = weights
        else:
            log('Initial weights already present.')

        if only_setup:
            return

        if self.cholesky is False:
            self.step = 0
            result = self.regressor.regress(model=self, log=log)
            return result  # True / False
        else:
            """
            This method would require solving to systems of linear equations.
            In the case of energies, we cannot operate in an atom-centered mode
            because we don't know a priori the energy per atom but per image.

            For forces is a different history because we do know the derivative
            of the energy with respect to atom positions (a per atom quantity).
            Therefore, obtaining weights with Cholesky decomposition would be
            the best for explicit-force training.
            """
            try:
                I_e = np.identity(self.size)
                K_e = self.kij.reshape(self.size, self.size)

                log('Starting Cholesky decomposition of kernel energy matrix to '
                'get upper triangular matrix.', tic='cholesky_energy_kernel')

                cholesky_U = cholesky((K_e + self.lamda * I_e))

                log('... Cholesky Decomposing finished in.',
                         toc='cholesky_energy_kernel')

                betas = np.linalg.solve(cholesky_U.T, self.energy_targets)
                weights = np.linalg.solve(cholesky_U, betas)
                p.weights['energy'] = weights

                if self.forcetraining is True:
                    log('Starting Cholesky decomposition of kernel force matrix to '
                    'get upper triangular matrix.', tic='cholesky_force_kernel')
                    force_weights = []
                    for i in range(3):
                        size = self.kernel_f_cholesky[i][0].size
                        I_f = np.identity(size)
                        K_f = self.kernel_f_cholesky[i].reshape(size, size)
                        cholesky_U = cholesky((K_f + self.lamda * I_f))
                        betas = np.linalg.solve(
                                   cholesky_U.T,
                                   self.force_targets[i]
                                   )
                        weights = np.linalg.solve(cholesky_U, betas)
                        force_weights.append(weights)
                    p.weights['forces'] = force_weights
                    log('... Cholesky Decomposing finished in.',
                             toc='cholesky_force_kernel')
                return True
            except np.linalg.linalg.LinAlgError:
                log('The kernel matrix seems to be singular. Add more\n'
                'noise to its diagonal elements by increasing the'
                'penalization term.'
                )
                return False
            except:
                return False

    def get_energy_kernel(self, trainingimages=None, fp_trainingimages=None,
                          only_features=False):
        """Local method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing information about the images. Note
            that you have to hash the images before passing them to this
            method.
        fp_trainingimages : object
            Fingerprints calculated using the trainingimages.
        only_features : bool
            If set to True, only the self.reference_features are built.

        Returns
        -------
        kij : list
            The kernel in form of a list.
        kernel_e : dictionary
            The kernel in a dictionary where keys are images' hashes.
        """
        # This creates a list containing all features in all images on the
        # training set.
        self.reference_features = []
        self.energy_targets = []

        hashes = list(hash_images(trainingimages).keys())

        for hash in hashes:
            if self.cholesky is False:
                for element, afp in fp_trainingimages[hash]:
                    afp = np.asarray(afp)
                    self.reference_features.append(afp)
            else:
                energy = trainingimages[hash].get_potential_energy()
                self.energy_targets.append(energy)
                afp = []
                for element, _afp in fp_trainingimages[hash]:
                    afp.append(_afp)

                self.reference_features.append(np.ravel(afp))

        self.kij = []

        if only_features is not True:
            for index, hash in enumerate(hashes):
                self.kernel_e[hash] = {}
                kernel = []

                if self.cholesky is False:
                    for index, (element, afp) in enumerate(
                            fp_trainingimages[hash]):
                        selfsymbol = element
                        selfindex = index
                        _kernel = self.kernel_matrix(
                                np.asarray(afp),
                                self.reference_features,
                                kernel=self.kernel
                                )
                        self.kernel_e[hash][(selfindex, selfsymbol)] = _kernel
                        kernel.append(_kernel)
                    self.kij.append(kernel)
                else:
                    afp = []
                    for element, _afp in fp_trainingimages[hash]:
                        afp.append(_afp)
                    _kernel = self.kernel_matrix(
                            np.ravel(afp),
                            self.reference_features,
                            kernel=self.kernel
                            )
                    self.kernel_e[hash] = _kernel
                    kernel.append(_kernel)
                    self.kij.append(kernel)

            self.kij = np.asarray(self.kij)

            return self.kernel_e

    def get_forces_kernel(self, trainingimages=None, t_descriptor=None,
                          only_features=False):
        """Method to get the kernel on the fly

        Parameters
        ----------
        trainingimages : object
            This is an ASE object containing the training set images. Note that
            images have to be hashed before passing them to this method.
        t_descriptor : object
            Descriptor object containing the fingerprintprimes from the
            training set.
        only_features : bool
            If set to True, only the self.reference_features are built.

        Returns
        -------
        self.kernel_f : dictionary
            Dictionary containing images hashes and kernels per atom.
        """

        forces_features_x = []
        forces_features_y = []
        forces_features_z = []

        hashes = list(hash_images(trainingimages).keys())
        fingerprintprimes = t_descriptor.fingerprintprimes
        fingerprint_name = t_descriptor.__class__.__name__

        self.force_features = {}

        for hash in hashes:
            self.force_features[hash] = {}
            image = trainingimages[hash]
            afps_prime_x = []
            afps_prime_y = []
            afps_prime_z = []

            # This loop assures that we are iterating from atom with index 0.
            for atom in image:
                selfsymbol = atom.symbol
                selfindex = atom.index
                self.force_features[hash][(selfindex, selfsymbol)] = {}

                fprime_sum_x,  fprime_sum_y, fprime_sum_z = 0., 0., 0.

                # Here we sum all different contributions of the derivatives of
                # the fingerprints
                for key in fingerprintprimes[hash].keys():
                    if (selfindex == key[0] and selfsymbol == key[1] and
                            key[-1] == 0):
                        fprime_sum_x += np.array(
                                    fingerprintprimes[hash][key])
                    elif (selfindex == key[0] and selfsymbol == key[1] and
                            key[-1] == 1):
                        fprime_sum_y += np.array(
                                    fingerprintprimes[hash][key])
                    elif (selfindex == key[0] and selfsymbol == key[1] and
                            key[-1] == 2):
                        fprime_sum_z += np.array(
                                    fingerprintprimes[hash][key])

                for component in range(3):
                    if component == 0:
                        afps_prime_x.append(fprime_sum_x)
                        self.force_features[hash][(
                            selfindex,
                            selfsymbol)][component] = fprime_sum_x
                    elif component == 1:
                        afps_prime_y.append(fprime_sum_y)
                        self.force_features[hash][(
                            selfindex,
                            selfsymbol)][component] = fprime_sum_y
                    else:
                        afps_prime_z.append(fprime_sum_z)
                        self.force_features[hash][(
                            selfindex,
                            selfsymbol)][component] = fprime_sum_z

            forces_features_x.append(afps_prime_x)
            forces_features_y.append(afps_prime_y)
            forces_features_z.append(afps_prime_z)

        # List containing all force features per component. Useful for
        # computing the kernels.
        self.reference_force_features = [
            list(itertools.chain.from_iterable(forces_features_x)),
            list(itertools.chain.from_iterable(forces_features_y)),
            list(itertools.chain.from_iterable(forces_features_z))
        ]

        if only_features is False:
            #if self.cholesky is True:
            self.force_targets = []
            self.kernel_f_cholesky = []
            kernel_x, targets_x = [], []
            kernel_y, targets_y = [], []
            kernel_z, targets_z = [], []

            for hash in hashes:
                image = trainingimages[hash]
                self.kernel_f[hash] = {}
                kernel = []

                #if self.cholesky is True:
                actual_forces = image.get_forces(apply_constraint=False)

                for atom in image:
                    selfsymbol = atom.symbol
                    selfindex = atom.index
                    self.kernel_f[hash][(selfindex, selfsymbol)] = {}
                    for component in range(3):
                        afp = self.force_features[hash][
                                (selfindex, selfsymbol)][component]
                        _kernel = self.kernel_matrix(
                                afp,
                                self.reference_force_features[component],
                                kernel=self.kernel
                                )
                        self.kernel_f[hash][
                                (selfindex, selfsymbol)][
                                        component] = _kernel

                        #if self.cholesky is True:
                        target = actual_forces[selfindex][component]
                        if component == 0:
                            kernel_x.append(_kernel)
                            targets_x.append(target)
                        elif component == 1:
                            kernel_y.append(_kernel)
                            targets_y.append(target)
                        elif component == 2:
                            kernel_z.append(_kernel)
                            targets_z.append(target)

            #if self.cholesky is True:
            self.kernel_f_cholesky = [
                    np.array(kernel_x),
                    np.array(kernel_y),
                    np.array(kernel_z)
                    ]
            self.force_targets = [
                    np.array(targets_x),
                    np.array(targets_y),
                    np.array(targets_z)
                    ]
            return self.kernel_f


    @property
    def forcetraining(self):
        """Returns True if forcetraining is turned on (as determined by
        examining the convergence criteria in the loss function), else
        returns False.
        """
        if self.lossfunction.parameters['force_coefficient'] is None:
            forcetraining = False
        elif self.lossfunction.parameters['force_coefficient'] > 0.:
            forcetraining = True
        return forcetraining

    @property
    def vector(self):
        """Access to get or set the model parameters (weights for each kernel)
        as a single vector, useful in particular for regression.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        if self.parameters['weights'] is None:
            return None
        p = self.parameters

        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(
                    p.weights,
                    weights_independent=self.weights_independent,
                    size=self.size
                    )
        return self.ravel.to_vector(weights=p.weights)

    @vector.setter
    def vector(self, vector):
        p = self.parameters

        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights)
        weights = self.ravel.to_dicts(vector)
        p['weights'] = weights

    def get_loss(self, vector):
        """Method to be called by the regression master.

        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the loss (cost) function.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """

        p = self.parameters
        if self.step == 0:
            filename = make_filename(self.parent.label,
                                     '-initial-parameters.amp')
            filename = self.parent.save(filename, overwrite=True)
        if self.checkpoints:
            if self.step % self.checkpoints == 0:
                self._log('Saving checkpoint data.')
                if self.checkpoints < 0:
                    path = os.path.join(self.parent.label + '-checkpoints')
                    if self.step == 0:
                        if not os.path.exists(path):
                            os.mkdir(path)
                    filename = os.path.join(path,
                                            '{}.amp'.format(int(self.step)))
                else:
                    filename = make_filename(self.parent.label,
                                             '-checkpoint.amp')
                self.parent.save(filename, overwrite=True)

        K_e = self.kij.reshape(self.size, self.size)
        K_f = self.kernel_f_cholesky
        loss = self.lossfunction.get_loss(
                vector,
                p.weights['energy'],
                K_e,
                p.weights['forces'],
                K_f ,
                lossprime=False
                )['loss']
        if hasattr(self, 'observer'):
            self.observer(self, vector, loss)
        self.step += 1
        return loss

    def get_lossprime(self, vector):
        """Method to be called by the regression master.

        Takes one and only one input, a vector of parameters.  Returns one
        output, the value of the derivative of the loss function with respect
        to model parameters.

        Parameters
        ----------
        vector : list
            Parameters of the regression model in the form of a list.
        """
        return self.lossfunction.get_loss(vector,
                                          lossprime=True)['dloss_dparameters']

    @property
    def lossfunction(self):
        """Allows the user to set a custom loss function.

        For example,
        >>> from amp.model import LossFunction
        >>> lossfxn = LossFunction(energy_tol=0.0001)
        >>> calc.model.lossfunction = lossfxn

        Parameters
        ----------
        lossfunction : object
            Loss function object, if at all desired by the user.
        """
        return self._lossfunction

    @lossfunction.setter
    def lossfunction(self, lossfunction):
        if hasattr(lossfunction, 'attach_model'):
            lossfunction.attach_model(self)  # Allows access to methods.
        self._lossfunction = lossfunction

    def calculate_atomic_energy(self, afp, index, symbol, hash=None,
                                fp_trainingimages=None, trainingimages=None,
                                kernel=None, sigma=None):
        """
        Given input to the KRR model, output (which corresponds to energy)
        is calculated about the specified atom. The sum of these for all
        atoms is the total energy (in atom-centered mode).

        Parameters
        ---------
        index: int
            Index of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).
        symbol : str
            Symbol of the atom for which atomic energy is calculated (only used
            in the atom-centered mode).
        hash : str
            hash of desired image to compute
        kernel : str
            The kernel to be computed in the case that Amp.load is used.
        sigma : float

        Returns
        -------
        atomic_amp_energy : float
            Atomic energy on atom with index=index.
        """
        if self.parameters.mode != 'atom-centered':
            raise AssertionError('calculate_atomic_energy should only be '
                                 ' called in atom-centered mode.')

        weights = self.parameters.weights

        if len(list(self.kernel_e.keys())) == 0 or hash not in self.kernel_e:
            kij_args = dict(
                    trainingimages=trainingimages,
                    fp_trainingimages=fp_trainingimages,
                    only_features=True
                    )

            # This is needed for both setting the size of parameters to
            # optimize and also to return the kernel for energies
            self.get_energy_kernel(**kij_args)
            kernel = self.kernel_matrix(
                            np.asarray(afp),
                            self.reference_features,
                            kernel=self.kernel,
                            sigma=sigma
                            )
            atomic_amp_energy = kernel.dot(weights['energy'][symbol])
        else:
            atomic_amp_energy = self.kernel_e[hash][
                        ((index, symbol))].dot(weights['energy'][symbol])
        return atomic_amp_energy

    def total_energy_from_cholesky(self, hash=None, fp_trainingimages=None,
                               trainingimages=None, kernel=None, sigma=None,
                               fingerprints=None):
        """
        Given input to the KRR model, output (which corresponds to energy)
        is calculated about the specified atom. The sum of these for all
        atoms is the total energy (in atom-centered mode).

        Parameters
        ---------
        hash : str
            hash of desired image to compute
        kernel : str
            The kernel to be computed in the case that Amp.load is used.
        sigma : float

        Returns
        -------
        atomic_amp_energy : float
            Atomic energy on atom with index=index.
        """
        if self.parameters.mode != 'atom-centered':
            raise AssertionError('calculate_atomic_energy should only be '
                                 ' called in atom-centered mode.')

        weights = self.parameters.weights

        if len(list(self.kernel_e.keys())) == 0 or hash not in self.kernel_e:
            kij_args = dict(
                    trainingimages=trainingimages,
                    fp_trainingimages=fp_trainingimages,
                    only_features=True
                    )

            # This is needed for both setting the size of parameters to
            # optimize and also to return the kernel for energies
            self.get_energy_kernel(**kij_args)
            afp = []
            for element, _afp in fingerprints:
                afp.append(_afp)

            kernel = self.kernel_matrix(
                            np.ravel(afp),
                            self.reference_features,
                            kernel=kernel,
                            sigma=sigma
                            )
            total_amp_energy = kernel.dot(weights['energy'])
        else:
            total_amp_energy = self.kernel_e[hash].dot(weights['energy'])
        return total_amp_energy

    def calculate_force(self, index, symbol, component, fingerprintprimes=None,
                        trainingimages=None, t_descriptor=None, sigma=None,
                        hash=None):
        """Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        Parameters
        ----------
        index : integer
            Index of central atom for which the atomic force will be computed.
        symbol : str
            Symbol of central atom for which the atomic force will be computed.
        component : int
            Direction of the force.
        fingerprintprimes : list
            List of fingerprint primes.
        trainingimages : list
            Object or list containing the training set. This is needed when
            performing predictions of unseen data.
        descriptor : object
            Object containing the information about fingerprints.
        hash : str
            Unique key for the image of interest.
        sigma : float

        Returns
        -------
        force : float
            Atomic force on Atom with index=index and symbol=symbol.
        """
        weights = self.parameters.weights
        key = index, symbol

        if len(list(self.kernel_f.keys())) == 0 or hash not in self.kernel_f:
            self.get_forces_kernel(
                    trainingimages=trainingimages,
                    t_descriptor=t_descriptor,
                    only_features=True
                    )

            fprime = 0
            for afp in fingerprintprimes:
                if (index == afp[0] and symbol == afp[1] and
                        component == afp[-1]):
                    fprime += np.array(fingerprintprimes[afp])

            features = self.reference_force_features[component]
            kernel = self.kernel_matrix(
                            fprime,
                            features,
                            kernel=self.kernel,
                            sigma=sigma
                            )
            if (self.weights_independent is True and self.cholesky is False):
                force = kernel.dot(weights['forces'][symbol][component])
            elif (self.weights_independent is False and self.cholesky is False):
                force = kernel.dot(weights['forces'][symbol])
        else:
            if (self.weights_independent is True and self.cholesky is False):
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][symbol][component]
                        )
            elif (self.weights_independent is False and self.cholesky is
                    False):
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][symbol]
                        )
        force *= -1.
        return force

    def forces_from_cholesky(self, index, symbol, component, fingerprintprimes=None,
                        trainingimages=None, t_descriptor=None, sigma=None,
                        hash=None):
        """Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        Parameters
        ----------
        index : integer
            Index of central atom for which the atomic force will be computed.
        symbol : str
            Symbol of central atom for which the atomic force will be computed.
        component : int
            Direction of the force.
        fingerprintprimes : list
            List of fingerprint primes.
        trainingimages : list
            Object or list containing the training set. This is needed when
            performing predictions of unseen data.
        descriptor : object
            Object containing the information about fingerprints.
        hash : str
            Unique key for the image of interest.
        sigma : float

        Returns
        -------
        force : float
            Atomic force on Atom with index=index and symbol=symbol.
        """
        weights = self.parameters.weights
        key = index, symbol

        if len(list(self.kernel_f.keys())) == 0 or hash not in self.kernel_f:
            self.get_forces_kernel(
                    trainingimages=trainingimages,
                    t_descriptor=t_descriptor,
                    only_features=True
                    )

            fprime = 0
            for afp in fingerprintprimes:
                if (index == afp[0] and symbol == afp[1] and
                        component == afp[-1]):
                    fprime += np.array(fingerprintprimes[afp])

            features = self.reference_force_features[component]
            kernel = self.kernel_matrix(
                            fprime,
                            features,
                            kernel=self.kernel,
                            sigma=sigma
                            )
            if (self.weights_independent is True and self.cholesky is True):
                force = kernel.dot(weights['forces'][component])
        else:
            if (self.weights_independent is True and self.cholesky is True):
                force = self.kernel_f[hash][key][component].dot(
                        weights['forces'][component]
                        )
        return force

    def kernel_matrix(self, feature, features, kernel='rbf', sigma=1.):
        """This method takes as arguments a feature vector and a string that refers
        to the kernel type used.

        Parameters
        ----------
        feature : list or numpy array
            Single feature.
        features : list or numpy array
            Column vector containing the fingerprints of all atoms in the
            training set.
        kernel : str
            Select the kernel to be used. Supported kernels are: 'linear',
            rbf', 'exponential, and 'laplacian'.
        sigma : float
            Length scale of the Gaussian in the case of RBF, exponential and
            laplacian kernels.

        Returns
        -------
        K : array
            The kernel matrix.

        Notes
        -----
        Kernels may differ a lot between them. The kernel_matrix method in this
        class contains algorithms to build the desired matrix. The computation
        of the kernel is done by auxiliary functions that are located at the
        end of the KRR class.
        """
        features = np.asarray(features)
        feature = np.asarray(feature)
        K = []

        call = {
                'exponential': exponential,
                'laplacian': laplacian,
                'rbf': rbf
                }

        if self.sigma is None:
            self.sigma = sigma

        if kernel == 'linear':
            for afp in features:
                K.append(linear(feature, afp))

        # All kernels in this control flow share the same structure
        elif (kernel == 'rbf' or kernel == 'laplacian' or
                kernel == 'exponential'):

            for afp in features:
                K.append(call[kernel](feature, afp, sigma=self.sigma))

        else:
            raise NotImplementedError('This kernel needs to be coded.')

        return np.asarray(K)


class Raveler(object):
    """Raveler class inspired by neuralnetwork.py

    Takes a weights dictionary created by KRR class and convert it into vector
    and back to dictionaries. This is needed for doing the optimization of the
    loss function.

    Parameters
    ----------
    weights : dict
        Dictionary containing weights per atom.
    size : int
        Number of elements in the dictionary.

    """
    def __init__(self, weights, weights_independent=None, size=None):
        self.count = 0
        self.weights_keys = []
        self.properties_keys = []
        self.size = size
        self.weights_independent = weights_independent

        for prop in weights.keys():
            self.properties_keys.append(prop)
            for key in weights[prop].keys():
                if prop is 'energy':
                    self.weights_keys.append(key)
                    self.count += len(weights[prop][key])
                elif prop is 'forces':
                    if self.weights_independent is True:
                        for component in range(3):
                            self.count += len(weights[prop][key][component])
                    else:
                        self.count += len(weights[prop][key])

    def to_vector(self, weights):
        """Convert weights dictionaries to one dimensional vectors.

        Parameters
        ----------
        weights : dict
            Dictionary of weights.

        Returns
        -------
        vector : ndarray
            One-dimensional weight vector to be used by the optimizer.
        """
        vector = []
        for prop in weights.keys():
            if prop is 'energy':
                for key in weights[prop].keys():
                    vector.append(weights[prop][key])
            elif prop is 'forces':
                if self.weights_independent is True:
                    for component in range(3):
                        for key in weights[prop].keys():
                            vector.append(weights[prop][key][component])
                else:
                    for key in weights[prop].keys():
                        vector.append(weights[prop][key])

        vector = np.ravel(vector)

        return vector

    def to_dicts(self, vector):
        """Convert vector of weights back into weights dictionaries.

        Parameters
        ----------
        vector : ndarray
            One-dimensional weight vector.

        Returns
        -------
        weights : dict
            Dictionary of weights.
        """

        assert len(vector) == self.count
        first = 0
        last = 0
        weights = OrderedDict()
        step = self.size

        for prop in self.properties_keys:
            weights[prop] = OrderedDict()
            if prop is 'energy':
                for k in self.weights_keys:
                    if k not in weights[prop].keys():
                        last += step
                        weights[prop][k] = vector[first:last]
                        first += step
            elif prop is 'forces':
                for k in self.weights_keys:
                    if (k not in weights[prop].keys() and
                            self.weights_independent is True):
                        weights[prop][k] = np.zeros((3, self.size))
                        for component in range(3):
                            last += step
                            weights[prop][k][
                                    component] = vector[first:last]
                            first += step
                    elif (k not in weights[prop].keys() and
                            self.weights_independent is False):
                        last += step
                        weights[prop][k] = vector[first:last]
                        first += step
        return weights


"""
Auxiliary functions to compute different kernels
"""


def linear(feature_i, feature_j):
    """ Compute a linear kernel """
    linear = np.dot(feature_i, feature_j)
    return linear


def rbf(feature_i, feature_j, sigma=1.):
    """ Compute the rbf (AKA Gaussian) kernel.  """
    rbf = np.exp(-(np.linalg.norm(feature_i - feature_j) ** 2.) /
            (2. * sigma ** 2.))
    return rbf


def exponential(feature_i, feature_j, sigma=1.):
    """ Compute the exponential kernel"""
    exponential = np.exp(-(np.linalg.norm(feature_i - feature_j)) /
                         (2. * sigma ** 2))
    return exponential


def laplacian(feature_i, feature_j, sigma=1.):
    """ Compute the laplacian kernel"""
    laplacian = np.exp(-(np.linalg.norm(feature_i - feature_j)) / sigma)
    return laplacian


def ravel_data(train_forces,
               mode,
               images,
               fingerprints,
               fingerprintprimes):
    """
    Reshapes data of images into lists.

    Parameters
    ----------
    train_forces : bool
        Determining whether forces are also trained or not.
    mode : str
        Can be either 'atom-centered' or 'image-centered'.
    images : list or str
        List of ASE atoms objects with positions, symbols, energies, and forces
        in ASE format. This is the training set of data. This can also be the
        path to an ASE trajectory (.traj) or database (.db) file. Energies can
        be obtained from any reference, e.g. DFT calculations.
    fingerprints : dict
        Dictionary with images hashs as keys and the corresponding fingerprints
        as values.
    fingerprintprimes : dict
        Dictionary with images hashs as keys and the corresponding fingerprint
        derivatives as values.
    """
    from ase.data import atomic_numbers

    actual_energies = [image.get_potential_energy(apply_constraint=False)
                       for image in images.values()]

    if mode == 'atom-centered':
        num_images_atoms = [len(image) for image in images.values()]
        atomic_numbers = [atomic_numbers[atom.symbol]
                          for image in images.values() for atom in image]

        def ravel_fingerprints(images,
                               fingerprints):
            """
            Reshape fingerprints of images into a list.
            """
            raveled_fingerprints = []
            elements = []
            for hash, image in images.items():
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
                            for image in images.values()]

    if train_forces is True:

        actual_forces = \
            [image.get_forces(apply_constraint=False)[index]
             for image in images.values() for index in range(len(image))]

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
                for hash, image in images.items():
                    for atom in image:
                        selfindex = atom.index
                        selfsymbol = atom.symbol
                        selfneighborindices = []
                        selfneighborsymbols = []
                        for key, derafp in fingerprintprimes[hash].items():
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
                         overfit,
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
                         model,
                         d):
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
    _fmodules.model_props.overfit = overfit
    _fmodules.model_props.train_forces = train_forces
    _fmodules.model_props.mode_signal = mode_signal
    if d is None:
        _fmodules.model_props.numericprime = False
    else:
        _fmodules.model_props.numericprime = True
        _fmodules.model_props.d = d

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
