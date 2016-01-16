
import numpy as np
from ase.calculators.calculator import Parameters

from ..utilities import ConvergenceOccurred, Logger, make_sublists


class LossFunction:
    """Basic cost function, which can be used by the model.get_cost_function
    method which is required in standard model classes.
    This version is pure python and thus will be slow compared to a fortran/parallel implementation.
    
    """

    def __init__(self, energy_tol=0.001, max_resid=0.001, cores=1,
                 raise_ConvergenceOccurred=True):
        p = self.parameters = Parameters(
            {'importname': '.model.LossFunction'})
        p['energy_tol'] = energy_tol
        p['max_resid'] = max_resid
        self._cores = cores
        self.raise_ConvergenceOccurred = raise_ConvergenceOccurred

    def attach_model(self, model, fingerprints=None, images=None,
                     initialize_sessions=True):
        """Attach the model to be used to the loss function. fingerprints and
        training images need not be supplied if they are already attached to
        the model via model.trainingparameters. Also initializes the server and sessions for
        parallel processing if initialize_sessions is True and multiple cores are
        specified."""
        self._model = model
        self.fingerprints = fingerprints
        self.images = images
        if fingerprints is None:
            self.fingerprints = model.trainingparameters.fingerprint.fingerprints
        if images is None:
            self.images = model.trainingparameters.trainingimages

        self.log = Logger(None)
        if hasattr(model, 'trainingparameters'):
            if 'log' in model.trainingparameters:
                self.log = model.trainingparameters['log']

        if initialize_sessions and self._cores != 1:
            import zmq
            import pxssh
            from socket import gethostname
            from getpass import getuser
            self.log(' Parallel processing.')
            context = zmq.Context()
            server = context.socket(zmq.REP)
            serverport = server.bind_to_random_port('tcp://*')
            serversocket = '%s:%s' % (gethostname(), serverport)
            self.log('  Established server at %s' % serversocket)

            module = self.__module__
            workercommand = ('python -m %s %%s %s' %
                             (module, serversocket))
            #FIXME/ap Lots of duplicate code here with utilities.Data
            # Probably should make into functions.

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
                self.log('  Session %i (%s): %s' % 
                    (process_id, workerhostname, ssh.before.strip()))
                return ssh

            # Create processes over SSH.
            self.log('  Establishging workers:')
            processes = []
            for workerhostname, nprocesses in self._cores.iteritems():
                n = len(processes)
                processes.extend([establish_ssh(_ + n) for _ in
                                  range(nprocesses)])

            self._sessions = {'master': server,
                              'workers': processes}



        p = self.parameters
        self.log(' Loss function attached to model. Convergence criteria:')
        self.log('  energy_tol: ' + str(p['energy_tol']))
        self.log('  max_resid: ' + str(p['max_resid']))
        self.log('\n  %12s  %12s' % ('EnergyLoss', 'MaxResid'))
        self.log('  %12s  %12s' % ('==========', '========'))

    def __call__(self, parametervector, complete_output=False):
        """Returns the current value of teh cost function for a given set of parameters,
        or, if the energy is less than the energy_tol raises a ConvergenceException.

        By default only returns the cost function (needed for optimizers). Can return
        more information like max_residual if complete_output is True.
        """
        #FIXME/ap If parallel, or fortran, implement that here.
        # There will need to be program instances (e.g., fortran
        # executables) that have the fingerprints, etc., already
        # stored and just calculate the cost function with the new
        # parameter vector.
        log = self.log

        if self._cores == 1:
            p = self.parameters
            self._model.set_vector(parametervector)
            costfxn = 0.
            max_residual = 0.
            for hash, image in self.images.iteritems():
                predicted = self._model.get_energy(self.fingerprints[hash])
                actual = image.get_potential_energy(apply_constraint=False)
                residual_per_atom = abs(predicted - actual) / len(image)
                if residual_per_atom > max_residual:
                    max_residual = residual_per_atom
                costfxn += residual_per_atom**2
            costfxn = costfxn / len(self.images)
        else:
            #FIXME/ap. There will have to be different procedures if it is the
            # first call or a later one. The channels will need to be kept open
            # such that it has the images, etc., passed only once.
            server = self._sessions['master']
            processes = self._sessions['workers']

            # Subdivide tasks.
            keys = make_sublists(self.images.keys(), len(processes))

            # All incoming requests will be dictionaries with three keys.
            # d['id']: process id number, as assigned when process created above.
            # d['subject']: what the message is asking for / telling you
            # d['data']: any data passed from the worker to the master in this msg.

            def process_parallels(vector):
                #FIXME/ap I have a problem with processes finishing then making
                # their next request. They need to wait for the process to cycle.
                finished = np.array([False]*len(processes))  # Flags for each process
                results = {'costfxn': 0., 'max_residual': 0.}
                while not finished.all():
                    message = server.recv_pyobj()
                    if message['subject'] == '<purpose>':
                        server.send_string('calculate_loss_function')
                    elif message['subject'] == '<request>':
                        request = message['data']  # Variable name.
                        if request == 'images':
                            subimages = {k:self.images[k] for k in keys[int(message['id'])]}
                            server.send_pyobj(subimages)
                        elif request == 'modelstring':
                            server.send_pyobj(self._model.tostring())
                        elif request == 'lossfunctionstring':
                            server.send_pyobj(self.parameters.tostring())
                        elif request == 'images': 
                            server.send_pyobj({k:images[k] for k in
                                               keys[int(message['id'])]})
                        elif request == 'fingerprints':
                            server.send_pyobj({k:self.fingerprints[k] for k in
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
                        results['costfxn'] += result['costfxn']
                        if result['max_residual'] > results['max_residual']:
                            results['max_residual'] = result['max_residual']
                        finished[int(message['id'])] = True
                return results


            results = process_parallels(parametervector)
            costfxn = results['costfxn']
            max_residual = results['max_residual']




        if self.raise_ConvergenceOccurred:
            converged = self.check_convergence(costfxn, max_residual)
            # Make sure first step is done in case of switching to fortran.
            if converged:
                self._model.set_vector(parametervector)
                raise ConvergenceOccurred()

        if complete_output is False:
            return costfxn
        else:
            return {'costfxn': costfxn,
                    'max_residual': max_residual, }

    def check_convergence(self, costfxn, max_residual):
        """Checks to see whether convergence is met; if it is, raises
        ConvergenceException to stop the optimizer."""
        p = self.parameters
        energyconverged = True
        maxresidconverged = True
        if p.energy_tol is not None:
            if costfxn > p.energy_tol:
                energyconverged = False
        if p.max_resid is not None:
            if max_residual > p.max_resid:
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

