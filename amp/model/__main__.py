"""Directly calling this module; apparently from another node.
Calls should come as

python -m amp.model id hostname:port

This session will then start a zmq session with that socket, labeling
itself with id. Instructions on what to do will come from the socket.
"""
import sys
import tempfile
import zmq
from amp.utilities import MessageDictionary, string2dict, Logger
from amp import importhelper
from amp.model import ravel_data, send_data_to_fortran


hostsocket = sys.argv[-1]
proc_id = sys.argv[-2]
msg = MessageDictionary(proc_id)

# Send standard lines to stdout signaling process started and where
# error is directed.
print('<amp-connect>')  # Signal that program started.
sys.stderr = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.stderr')
print('Log and stderr written to %s<stderr>' % sys.stderr.name)

# Also send logger output to stderr to aid in debugging.
log = Logger(file=sys.stderr)

# Establish client session via zmq; find purpose.
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://%s' % hostsocket)
socket.send_pyobj(msg('<purpose>'))
purpose = socket.recv_string()

if purpose == 'calculate_loss_function':
    # Request variables.
    socket.send_pyobj(msg('<request>', 'fortran'))
    fortran = socket.recv_pyobj()
    socket.send_pyobj(msg('<request>', 'modelstring'))
    modelstring = socket.recv_pyobj()
    dictionary = string2dict(modelstring)
    Model = importhelper(dictionary.pop('importname'))
    log('Model received:')
    log(str(dictionary))
    model = Model(fortran=fortran, **dictionary)
    model.log = log
    log('Model set up.')

    socket.send_pyobj(msg('<request>', 'args'))
    args = socket.recv_pyobj()
    d = args['d']
    socket.send_pyobj(msg('<request>', 'lossfunctionstring'))
    lossfunctionstring = socket.recv_pyobj()
    dictionary = string2dict(lossfunctionstring)
    log(str(dictionary))
    LossFunction = importhelper(dictionary.pop('importname'))
    lossfunction = LossFunction(parallel={'cores': 1},
                                raise_ConvergenceOccurred=False,
                                d=d, **dictionary)
    log('Loss function set up.')

    images = None
    socket.send_pyobj(msg('<request>', 'images'))
    images = socket.recv_pyobj()
    log('Images received.')

    fingerprints = None
    socket.send_pyobj(msg('<request>', 'fingerprints'))
    fingerprints = socket.recv_pyobj()
    log('Fingerprints received.')

    fingerprintprimes = None
    socket.send_pyobj(msg('<request>', 'fingerprintprimes'))
    fingerprintprimes = socket.recv_pyobj()
    log('Fingerprintprimes received.')

    # Set up local loss function.
    lossfunction.attach_model(model,
                              fingerprints=fingerprints,
                              fingerprintprimes=fingerprintprimes,
                              images=images)
    log('Images, fingerprints, and fingerprintprimes '
        'attached to the loss function.')

    if model.fortran:
        try:
            from .. import fmodules
        except ImportError:
            fmodules = None
        log('fmodules: %s' % str(fmodules))

        mode = model.parameters.mode
        energy_coefficient = lossfunction.parameters.energy_coefficient
        force_coefficient = lossfunction.parameters.force_coefficient
        overfit = lossfunction.parameters.overfit
        if force_coefficient == 0.:
            train_forces = False
        else:
            train_forces = True

        # FIXME: Should be corrected for image-centered:
        if mode == 'atom-centered':
            num_atoms = None

        (actual_energies, actual_forces, elements, atomic_positions,
         num_images_atoms, atomic_numbers, raveled_fingerprints, num_neighbors,
         raveled_neighborlists, raveled_fingerprintprimes) = (None,) * 10

        value = ravel_data(train_forces,
                           mode,
                           images,
                           fingerprints,
                           fingerprintprimes,)

        if mode == 'image-centered':
            if not train_forces:
                (actual_energies, atomic_positions) = value
            else:
                (actual_energies, actual_forces, atomic_positions) = value
        else:
            if not train_forces:
                (actual_energies, elements, num_images_atoms, atomic_numbers,
                 raveled_fingerprints) = value
            else:
                (actual_energies, actual_forces, elements, num_images_atoms,
                 atomic_numbers, raveled_fingerprints, num_neighbors,
                 raveled_neighborlists, raveled_fingerprintprimes) = value
        log('Data reshaped into lists to be sent to fmodules.')

        num_images = len(images)

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
                             model,
                             lossfunction.d)
        log('Data sent to fmodules.')

    if model.fortran:
        log('fmodules will be used to evaluate loss function.')
    else:
        log('Fortran will not be used to evaluate loss function.')
    # Now wait for parameters, and send the component of the loss function.
    while True:
        socket.send_pyobj(msg('<request>', 'parameters'))
        parameters = socket.recv_pyobj()
        if parameters == '<stop>':
            if model.fortran:
                fmodules.deallocate_variables()
            break
        elif parameters == '<continue>':
            # Master is waiting for other workers to finish.
            # Any more elegant way
            # to do this without opening another comm channel?
            # or having a thread for each process?
            pass
        else:
            socket.send_pyobj(msg('<request>', 'args'))
            args = socket.recv_pyobj()
            lossprime = args['lossprime']
            if model.fortran:
                # AKh: Right now, for each optimizer call, this function is
                # called. This is not needed. We already have fprime and
                # do not need to call this function when fprime is wanted
                # (except in the first step, because optimizer first call
                # fprime). This whole issue does not apply to fortran=False
                # and is fine there.
                (loss, dloss_dparameters, energy_loss, force_loss,
                 energy_maxresid, force_maxresid) = \
                    fmodules.calculate_loss(parameters=parameters,
                                            num_parameters=len(parameters),
                                            lossprime=lossprime)
                output = {'loss': loss,
                          'dloss_dparameters': dloss_dparameters,
                          'energy_loss': energy_loss,
                          'force_loss': force_loss,
                          'energy_maxresid': energy_maxresid,
                          'force_maxresid': force_maxresid, }
            else:
                output = lossfunction.get_loss(parameters,
                                               lossprime=lossprime)

            socket.send_pyobj(msg('<result>', output))
            socket.recv_string()

else:
    raise NotImplementedError('Purpose "%s" unknown.' % purpose)
