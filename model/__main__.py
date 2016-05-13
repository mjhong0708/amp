"""Directly calling this module; apparently from another node.
Calls should come as

python -m amp.model id hostname:port

This session will then start a zmq session with that socket, labeling
itself with id. Instructions on what to do will come from the socket.
"""
import sys
import tempfile
import zmq
from ..utilities import MessageDictionary, string2dict, Logger
from amp import importhelper


hostsocket = sys.argv[-1]
proc_id = sys.argv[-2]
msg = MessageDictionary(proc_id)

# Send standard lines to stdout signaling process started and where
# error is directed.
print('<amp-connect>')  # Signal that program started.
sys.stderr = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.stderr')
print('stderr written to %s<stderr>' % sys.stderr.name)

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
    socket.send_pyobj(msg('<request>', 'modelstring'))
    modelstring = socket.recv_pyobj()
    d = string2dict(modelstring)
    Model = importhelper(d.pop('importname'))
    log(str(d))
    model = Model(**d)
    model.log = log

    socket.send_pyobj(msg('<request>', 'lossfunctionstring'))
    lossfunctionstring = socket.recv_pyobj()
    d = string2dict(lossfunctionstring)
    sys.stderr.write(str(d))
    LossFunction = importhelper(d.pop('importname'))
    lossfunction = LossFunction(cores=1,
                                raise_ConvergenceOccurred=False, **d)

    socket.send_pyobj(msg('<request>', 'images'))
    images = socket.recv_pyobj()

    socket.send_pyobj(msg('<request>', 'fingerprints'))
    fingerprints = socket.recv_pyobj()

    socket.send_pyobj(msg('<request>', 'fingerprintprimes'))
    fingerprintprimes = socket.recv_pyobj()

    # Set up local loss function.
    lossfunction.attach_model(model, fingerprints=fingerprints,
                              fingerprintprimes=fingerprintprimes,
                              images=images)

    # Now wait for parameters, and send the component of the cost
    # function.
    while True:
        socket.send_pyobj(msg('<request>', 'parameters'))
        parameters = socket.recv_pyobj()
        if parameters == '<stop>':
            break
        elif parameters == '<continue>':
            # Master is waiting for other workers to finish.
            # Any more elegant way
            # to do this without opening another comm channel?
            # or having a thread for each process?
            pass
        else:
            output = lossfunction(parameters, complete_output=True)
            socket.send_pyobj(msg('<result>', output))
            socket.recv_string()

else:
    raise NotImplementedError('purpose %s unknown.' % purpose)
