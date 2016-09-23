#!/usr/bin/env python

import numpy as np
import hashlib
import time
import os
import copy
import math
import random
import signal
import pickle
from ase import io as aseio
from ase.parallel import paropen
from ase.db import connect
import shelve
from datetime import datetime
from threading import Thread
from getpass import getuser


# Parallel processing ########################################################

def assign_cores(cores, log=None):
    """Tries to guess cores from environment. If fed a log object, will write
    it's progress."""
    log = Logger(None) if log is None else log

    def fail(q):
        msg = ('Auto core detection is either not set up or not working for'
               ' your version of %s. You are invited to submit a patch to '
               'return a dictionary of the form {nodename: ncores} for this'
               ' batching system. The environment contents were dumped to '
               'the log file.')
        log('Environment dump:')
        for key, value in os.environ.iteritems():
            log('%s: %s' % (key, value))
        raise NotImplementedError(msg % q)

    def success(q, cores, log):
        log('Parallel configuration determined from environment for %s:' % q)
        for key, value in cores.iteritems():
            log('  %s: %i' % (key, value))

    if cores is not None:
        q = '<user-specified>'
        if cores == 1:
            log('Serial operation on one core specified.')
            return cores
        else:
            try:
                cores = int(cores)
            except TypeError:
                cores = cores
                success(q, cores, log)
                return cores
            else:
                cores = {'localhost': cores}
                success(q, cores, log)
                return cores

    if 'SLURM_NODELIST' in os.environ.keys():
        q = 'SLURM'
        nnodes = int(os.environ['SLURM_NNODES'])
        nodes = os.environ['SLURM_NODELIST']
        taskspernode = int(os.environ['SLURM_TASKS_PER_NODE'])
        if nnodes == 1:
            cores = {nodes: taskspernode}
        else:
            fail(q)
    elif 'PBS_NODEFILE' in os.environ.keys():
        fail(q='PBS')
    elif 'LOADL_PROCESSOR_LIST' in os.environ.keys():
        fail(q='LOADL')
    elif 'PE_HOSTFILE' in os.environ.keys():
        fail(q='SGE')
    else:
        import multiprocessing
        ncores = multiprocessing.cpu_count()
        cores = {'localhost': ncores}
        log('No queuing system detected; single machine assumed.')
        q = '<single machine>'
    success(q, cores, log)
    return cores


class MessageDictionary:

    """Standard container for all messages (typically requests, via
    zmq.context.socket.send_pyobj) sent from the workers to the master.
    This returns a simple dictionary. This is roughly email format.
    Initialize with process id (e.g., 'from'). Call with subject and data
    (body).
    """

    def __init__(self, process_id):
        self._process_id = process_id

    def __call__(self, subject, data=None):
        d = {'id': self._process_id,
             'subject': subject,
             'data': data}
        return d


def make_sublists(masterlist, n):
    """Randomly divides the masterlist into n sublists of roughly
    equal size. The intended use is to divide a keylist and assign
    keys to each task in parallel processing. This also destroys
    the masterlist (to save some memory)."""
    np.random.shuffle(masterlist)
    N = len(masterlist)
    sublist_lengths = [
        N // n if _ >= (N % n) else N // n + 1 for _ in range(n)]
    sublists = []
    for sublist_length in sublist_lengths:
        sublists.append([masterlist.pop() for _ in xrange(sublist_length)])
    return sublists


class EstablishSSH(Thread):

    """A thread to start a new SSH session. Starting via threads allows all
    sessions to start simultaneously, rather than waiting on one another.
    Access its created session with self.ssh.
    """

    def __init__(self, process_id, workerhostname, workercommand, log):
        self._process_id = process_id
        self._workerhostname = workerhostname
        self._workercommand = workercommand
        self._log = log
        Thread.__init__(self)

    def run(self):
        pxssh = importer('pxssh')
        ssh = pxssh.pxssh()
        ssh.login(self._workerhostname, getuser())
        ssh.sendline(self._workercommand % self._process_id)
        ssh.expect('<amp-connect>')
        ssh.expect('<stderr>')
        self._log('  Session %i (%s): %s' %
                  (self._process_id, self._workerhostname, ssh.before.strip()))
        self.ssh = ssh


# Data and logging ###########################################################


class SQLiteDB:
    """Replacement to shelve.
    Meant to mimic the same commands as shelve has so it is compatible with
    Data class. If sqlitedict is not available, it falls back to shelve.
    Note this calls yet another class, SQD, below. This could
    be cleaned up a bit.
    """
    def __init__(self, maxretries=100, retrypause=10.0):
        self.use_shelve = False
        try:
            import sqlitedict
        except ImportError:
            self.use_shelve = True
        self.maxretries = maxretries
        self.retrypause = retrypause

    def open(self, filename, flag=None):
        if self.use_shelve:
            self.d = shelve.open(filename, flag=flag)
        else:
            from sqlitedict import SqliteDict
            from sqlite3 import OperationalError
            #self.d = SqliteDict(filename, autocommit=True)
            class SQD(SqliteDict):
                def __init__(self, filename, autocommit,
                             maxretries, retrypause):
                    self.maxretries = maxretries
                    self.retrypause = retrypause
                    SqliteDict.__init__(self, filename, autocommit=autocommit)
                def __setitem__(self, key, value):
                    tries = 0
                    success = False
                    while not success:
                        try:
                            SqliteDict.__setitem__(self, key, value)
                        except OperationalError:
                            tries += 1
                            time.sleep(self.retrypause)
                            if tries >= self.maxretries:
                                raise
                        else:
                            success = True
                def __getitem__(self, key):
                    tries = 0
                    success = False
                    while not success:
                        try:
                            return SqliteDict.__getitem__(self, key)
                        except OperationalError:
                            tries += 1
                            time.sleep(self.retrypause)
                            if tries >= self.maxretries:
                                raise
                        else:
                            success = True
                def close(self):
                    tries = 0
                    success = False
                    while not success:
                        try:
                            SqliteDict.close(self)
                        except OperationalError:
                            tries += 1
                            time.sleep(self.retrypause)
                            if tries >= self.maxretries:
                                raise
                        else:
                            success = True

            self.d = SQD(filename, autocommit=True,
                         maxretries=self.maxretries, retrypause=self.retrypause)

        return self.d


class Data:

    """
    Serves as a container (dictionary-like) for (key, value) pairs that
    also serves to calculate them.
    Works by default with python's shelve module, but something that is
    built to share the same commands as shelve will work fine; just specify
    this in dbinstance.
    Designed to hold things like neighborlists, which have a hash, value
    format.
    This will work like a dictionary in that items can be accessed with
    data[key], but other advanced dictionary functions should be accessed
    with through the .d attribute:
    >>> data = Data(...)
    >>> data.open()
    >>> keys = data.d.keys()
    >>> values = data.d.values()
    """

    def __init__(self, filename, db=SQLiteDB(), calculator=None):
        self.calc = calculator
        self.db = db
        self.filename = filename
        self.d = None

    def calculate_items(self, images, cores=1, log=None):
        """Calculates the data value with 'calculator' for the specified
        images. images is a dictionary, and the same keys will be used for
        the current database."""
        if log is None:
            log = Logger(None)
        if self.d is not None:
            self.d.close()
            self.d = None
        log(' Data stored in file %s.' % self.filename)
        d = self.db.open(self.filename, 'c')
        calcs_needed = list(set(images.keys()).difference(d.keys()))
        dblength = len(d)
        d.close()
        log(' File exists with %i total images, %i of which are needed.' %
            (dblength, len(images) - len(calcs_needed)))
        log(' %i new calculations needed.' % len(calcs_needed))
        if len(calcs_needed) == 0:
            return
        if cores == 1:
            d = self.db.open(self.filename, 'c')  # FIXME/ap Need a lock?
            for key in calcs_needed:
                d[key] = self.calc.calculate(images[key], key)
            d.close()  # Necessary to get out of write mode and unlock?
            log(' Calculated %i new images.' % len(calcs_needed))
        else:
            import zmq
            from socket import gethostname
            pxssh = importer('pxssh')
            log(' Parallel processing.')
            module = self.calc.__module__
            globals = self.calc.globals
            keyed = self.calc.keyed
            serverhostname = gethostname()

            # Establish server session.
            context = zmq.Context()
            server = context.socket(zmq.REP)
            port = server.bind_to_random_port('tcp://*')
            serversocket = '%s:%s' % (serverhostname, port)
            log(' Established server at %s.' % serversocket)

            workercommand = 'python -m %s %%s %s' % (module, serversocket)

            # Create processes over SSH.
            # 'processes' contains links to the actual processes;
            # 'threads' is only used here to start all the SSH connections
            # simultaneously.
            log(' Establishing worker sessions.')
            processes = []
            threads = []  # Only used to start processes.
            for workerhostname, nprocesses in cores.iteritems():
                for pid in range(len(threads), len(threads) + nprocesses):
                    threads.append(EstablishSSH(pid,
                                                workerhostname,
                                                workercommand, log))
            for thread in threads:
                thread.start()
                time.sleep(0.5)
            for thread in threads:
                thread.join()
            for thread in threads:
                processes.append(thread.ssh)

            # All incoming requests will be dictionaries with three keys.
            # d['id']: process id number, assigned when process created above.
            # d['subject']: what the message is asking for / telling you
            # d['data']: optional data passed from the worker.

            keys = make_sublists(calcs_needed, len(processes))
            results = {}

            active = 0  # count of processes actively calculating
            log(' Parallel calculations starting...', tic='parallel')
            while True:
                message = server.recv_pyobj()
                if message['subject'] == '<purpose>':
                    server.send_pyobj(self.calc.parallel_command)
                    active += 1
                elif message['subject'] == '<request>':
                    request = message['data']  # Variable name.
                    if request == 'images':
                        server.send_pyobj({k: images[k] for k in
                                           keys[int(message['id'])]})
                    elif request in keyed:
                        server.send_pyobj({k: keyed[request][k] for k in
                                           keys[int(message['id'])]})
                    else:
                        server.send_pyobj(globals[request])
                elif message['subject'] == '<result>':
                    result = message['data']
                    server.send_string('meaningless reply')
                    active -= 1
                    log('  Process %s returned %i results.' %
                        (message['id'], len(result)))
                    results.update(result)
                elif message['subject'] == '<info>':
                    print('  %s' % message['data'])
                    server.send_string('meaningless reply')
                if active == 0:
                    break
            log('  %i new results.' % len(results))
            log(' ...parallel calculations finished.', toc='parallel')
            log(' Adding new results to database.')
            d = self.db.open(self.filename, 'c')  # FIXME/ap Need a lock?
            d.update(results)
            d.close()  # Necessary to get out of write mode and unlock?

        self.d = None

    def __getitem__(self, key):
        self.open()
        return self.d[key]

    def close(self):
        """Safely close the database."""
        if self.d:
            self.d.close()
        self.d = None

    def open(self, mode='r'):
        """Open the database connection with mode specified."""
        if self.d is None:
            self.d = self.db.open(self.filename, mode)

    def __del__(self):
        self.close()


class Logger:

    """
    Logger that can also deliver timing information.

    :param file: File object or path to the file to write to.
                     Or set to None for a logger that does nothing.
    :type file: str
    """

    def __init__(self, file):
        if file is None:
            self.file = None
            return
        if isinstance(file, str):
            self.filename = file
            file = paropen(file, 'a')
        self.file = file
        self.tics = {}

    def tic(self, label=None):
        """
        Start a timer.

        :param label: Label for managing multiple timers.
        :type label: str
        """
        if self.file is None:
            return
        if label:
            self.tics[label] = time.time()
        else:
            self._tic = time.time()

    def __call__(self, message, toc=None, tic=False):
        """
        Writes message to the log file.

        :param message: Message to be written.
        :type message: str
        :param toc: If toc=True or toc=label, it will append timing information
                    in minutes to the timer.
        :type toc: bool or str
        :param tic: If tic=True or tic=label, will start the generic timer
                    or a timer associated with label. Equivalent to
                    self.tic(label).
        :type tic: bool or str
        """
        if self.file is None:
            return
        dt = ''
        if toc:
            if toc is True:
                tic = self._tic
            else:
                tic = self.tics[toc]
            dt = (time.time() - tic) / 60.
            dt = ' %.1f min.' % dt
        if self.file.closed:
            self.file = paropen(self.filename, 'a')
        self.file.write(message + dt + '\n')
        self.file.flush()
        if tic:
            if tic is True:
                self.tic()
            else:
                self.tic(label=tic)


def make_filename(label, base_filename):
    """
    Creates a filename from the label and the base_filename which should be
    a string. Returns None if label is None; that is, it only saves output
    if a label is specified.

    :param label: Prefix.
    :type label: str
    :param base_filename: Basic name of the file.
    :type base_filename: str
    """

    if label is None:
        return None
    if not label:
        filename = base_filename
    else:
        filename = os.path.join(label + base_filename)

    return filename


# Images and hasing ##########################################################

def hash_image(atoms):
    """
    Creates a unique signature for a particular ASE atoms object.
    This is used to check whether an image has been seen before.
    This is just an md5 hash of a string representation of the atoms
    object.

    :param atoms: ASE atoms object.
    :type atoms: ASE dict

    :returns: Hash key of 'atoms'.
    """
    string = str(atoms.pbc)
    for number in atoms.cell.flatten():
        string += '%.15f' % number
    string += str(atoms.get_atomic_numbers())
    for number in atoms.get_positions().flatten():
        string += '%.15f' % number

    md5 = hashlib.md5(string)
    hash = md5.hexdigest()
    return hash


def hash_images(images, log=None, ordered=False):
    """
    Converts input images -- which may be a list, a trajectory file, or a
    database -- into a dictionary indexed by their hashes. Returns this
    dictionary. If ordered is True, returns an OrderedDict. When duplicate
    images are encountered (based on encountering an identical hash), a
    warning is written to the logfile. The number of duplicates of each
    image can be accessed by examinging dict_images.metadata['duplicates'],
    where dict_images is the returned dictionary.
    """
    if log is None:
        log = Logger(None)
    if images is None:
        return
    elif hasattr(images, 'keys'):
        return images  # Apparently already hashed.
    else:
        # Need to be hashed, and possibly read from file.
        if isinstance(images, str):
            log('Attempting to read images from file %s.' %
                images)
            extension = os.path.splitext(images)[1]
            from ase import io
            if extension == '.traj':
                images = io.Trajectory(images, 'r')
            elif extension == '.db':
                images = [row.toatoms() for row in
                          connect(images, 'db').select(None)]

        # images converted to dictionary form; key is hash of image.
        log('Hashing images...', tic='hash')
        dict_images = MetaDict()
        dict_images.metadata['duplicates'] = {}
        dup = dict_images.metadata['duplicates']
        if ordered is True:
            from collections import OrderedDict
            dict_images = OrderedDict()
        for image in images:
            hash = hash_image(image)
            if hash in dict_images.keys():
                log('Warning: Duplicate image (based on identical hash).'
                    ' Was this expected? Hash: %s' % hash)
                if hash in dup.keys():
                    dup[hash] += 1
                else:
                    dup[hash] = 2
            dict_images[hash] = image
        log(' %i unique images after hashing.' % len(dict_images))
        log('...hashing completed.', toc='hash')
        return dict_images


def randomize_images(images, fraction=0.8):
    """
    Randomly assigns 'fraction' of the images to a training set and
    (1 - 'fraction') to a test set. Returns two lists of ASE images.

    :param images: List of ASE atoms objects in ASE format. This can also be
                   the path to an ASE trajectory (.traj) or database (.db)
                   file.
    :type images: list or str
    :param fraction: Portion of train_images to all images.
    :type fraction: float

    :returns: Lists of train and test images.
    """
    file_opened = False
    if type(images) == str:
        extension = os.path.splitext(images)[1]
        if extension == '.traj':
            images = aseio.Trajectory(images, 'r')
        elif extension == '.db':
            images = aseio.read(images)
        file_opened = True

    trainingsize = int(fraction * len(images))
    testsize = len(images) - trainingsize
    testindices = []
    while len(testindices) < testsize:
        next = np.random.randint(len(images))
        if next not in testindices:
            testindices.append(next)
    testindices.sort()
    trainindices = [index for index in range(len(images)) if index not in
                    testindices]
    train_images = [images[index] for index in trainindices]
    test_images = [images[index] for index in testindices]
    if file_opened:
        images.close()
    return train_images, test_images

# Custom exceptions ##########################################################
# FIXME/ap Make sure these are all still imported somewhere.


class FingerprintsError(Exception):

    """ Error in case functional form of fingerprints has changed."""
    pass


class ConvergenceOccurred(Exception):

    """ Kludge to decide when scipy's optimizers are complete."""
    pass


class TrainingConvergenceError(Exception):

    """Error to be raised if training does not converge."""
    pass


class ExtrapolateError(Exception):

    """Error class in the case of extrapolation."""
    pass


class UntrainedError(Exception):

    """Error class in the case of unsuccessful training."""
    pass


# Miscellaneous ##############################################################

def string2dict(text):
    """Converts a string into a dictionary. Basically just calls `eval` on
    it, but supplies words like OrderedDict and matrix."""
    try:
        dictionary = eval(text)
    except NameError:
        from collections import OrderedDict
        from numpy import array, matrix
        dictionary = eval(text)
    return dictionary


def now(with_utc=False):
    """
    :returns: String of current time.'
    """
    local = datetime.now().isoformat().split('.')[0]
    utc = datetime.utcnow().isoformat().split('.')[0]
    if with_utc:
        return '%s (%s UTC)' % (local, utc)
    else:
        return local


logo = """
   oo      o       o   oooooo
  o  o     oo     oo   o     o
 o    o    o o   o o   o     o
o      o   o  o o  o   o     o
oooooooo   o   o   o   oooooo
o      o   o       o   o
o      o   o       o   o
o      o   o       o   o
"""


def importer(modulename):
    """Handles strange import cases, like pxssh which might show
    up in pexpect or pxssh."""

    if modulename == 'pxssh':
        try:
            import pxssh
        except ImportError:
            try:
                from pexpect import pxssh
            except ImportError:
                raise ImportError('pexpect not found!')
        return pxssh


def perturb_parameters(filename, images, d=0.0001, overwrite=False, **kwargs):
    """Returns the plot of loss function in terms of perturbed parameters.
    Takes the name of ".amp" file and images. Any other keyword taken by the
    Amp calculator can be fed to this class also.
    """

    from . import Amp
    from amp.descriptor.gaussian import Gaussian
    from amp.model.neuralnetwork import NeuralNetwork
    from amp.model import LossFunction

    calc = Amp(descriptor=Gaussian(),
               model=NeuralNetwork(),
               **kwargs)
    calc = calc.load(filename=filename)

    filename = make_filename(calc.label, '-perturbed-parameters.pdf')
    if (not overwrite) and os.path.exists(filename):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.' % filename)

    images = hash_images(images)

    # FIXME: AKh: Should read from filename, after it is saved.
    train_forces = True
    calculate_derivatives = train_forces
    calc.descriptor.calculate_fingerprints(
            images=images,
            cores=calc.cores,
            log=calc.log,
            calculate_derivatives=calculate_derivatives)

    vector = calc.model.vector.copy()

    # FIXME: AKh: Should read from filename, after it is saved.
    lossfunction = LossFunction(energy_coefficient=1.0,
                                force_coefficient=0.05,
                                cores=calc.cores,
                                )
    calc.model.lossfunction = lossfunction

    # Set up local loss function.
    lossfunction.attach_model(
            calc.model,
            fingerprints=calc.descriptor.fingerprints,
            fingerprintprimes=calc.descriptor.fingerprintprimes,
            images=images)

    originalloss = calc.model.get_loss(vector,
                                       complete_output=False)

    calc.log('\n Perturbing parameters...', tic='perturb')

    allparameters = []
    alllosses = []
    num_parameters = len(vector)

    for count in range(num_parameters):
        calc.log('parameter %i out of %i' % (count + 1, num_parameters))
        parameters = []
        losses = []
        # parameter is perturbed -d and loss function calculated.
        vector[count] -= d
        parameters.append(vector[count])
        perturbedloss = calc.model.get_loss(vector, complete_output=False)
        losses.append(perturbedloss)

        vector[count] += d
        parameters.append(vector[count])
        losses.append(originalloss)
        # parameter is perturbed +d and loss function calculated.
        vector[count] += d
        parameters.append(vector[count])
        perturbedloss = calc.model.get_loss(vector, complete_output=False)
        losses.append(perturbedloss)

        allparameters.append(parameters)
        alllosses.append(losses)
        # returning back to the original value.
        vector[count] -= d

    calc.log('...parameters perturbed and loss functions calculated',
             toc='perturb')

    calc.log('Plotting loss function vs perturbed parameters...',
             tic='plot')

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rcParams
    from matplotlib import pyplot
    from matplotlib.backends.backend_pdf import PdfPages
    rcParams.update({'figure.autolayout': True})

    with PdfPages(filename) as pdf:
        count = 0
        for parameter in vector:
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            ax.plot(allparameters[count],
                    alllosses[count],
                    marker='o', linestyle='--', color='b',)

            xmin = allparameters[count][0] - \
                0.1 * (allparameters[count][-1] - allparameters[count][0])
            xmax = allparameters[count][-1] + \
                0.1 * (allparameters[count][-1] - allparameters[count][0])
            ymin = min(alllosses[count]) - \
                0.1 * (max(alllosses[count]) - min(alllosses[count]))
            ymax = max(alllosses[count]) + \
                0.1 * (max(alllosses[count]) - min(alllosses[count]))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

            ax.set_xlabel('parameter no %i' % count)
            ax.set_ylabel('loss function')
            pdf.savefig(fig)
            pyplot.close(fig)
            count += 1

    calc.log(' ...loss functions plotted.', toc='plot')


# Amp Simulated Annealer ######################################################


class Annealer(object):

    """
    Inspired by the simulated annealing implementation of
    Richard J. Wagner <wagnerr@umich.edu> and
    Matthew T. Perry <perrygeo@gmail.com> at
    https://github.com/perrygeo/simanneal.

    Performs simulated annealing by calling functions to calculate loss and
    make moves on a state.  The temperature schedule for annealing may be
    provided manually or estimated automatically.

    Can be used by something like:

    >>> from amp import Amp
    >>> from amp.descriptor.gaussian import Gaussian
    >>> from amp.model.neuralnetwork import NeuralNetwork
    >>> calc = Amp(descriptor=Gaussian(), model=NeuralNetwork())

    which will initialize tha calc object as usual, and then

    >>> from amp.utilities import Annealer
    >>> Annealer(calc=calc, images=images)

    which will perform simulated annealing global search in parameters space,
    and finally

    >>> calc.train(images=images)

    for gradient descent optimization.
    """

    Tmax = 20.0             # Max (starting) temperature
    Tmin = 2.5              # Min (ending) temperature
    steps = 10000           # Number of iterations
    updates = steps / 200   # Number of updates (an update prints to log)
    copy_strategy = 'copy'
    user_exit = False
    save_state_on_exit = False

    def __init__(self, calc, images,
                 Tmax=None, Tmin=None, steps=None, updates=None):
        if Tmax is not None:
            self.Tmax = Tmax
        if Tmin is not None:
            self.Tmin = Tmin
        if steps is not None:
            self.steps = steps
        if updates is not None:
            self.updates = updates
        self.calc = calc

        self.calc.log('\nAmp simulated annealer started. ' + now() + '\n')
        self.calc.log('Descriptor: %s' %
                      self.calc.descriptor.__class__.__name__)
        self.calc.log('Model: %s' % self.calc.model.__class__.__name__)

        images = hash_images(images, log=self.calc.log)

        self.calc.log('\nDescriptor\n==========')
        # Derivatives of fingerprints need to be calculated if train_forces is
        # True.
        calculate_derivatives = True
        self.calc.descriptor.calculate_fingerprints(
                images=images,
                cores=self.calc.cores,
                log=self.calc.log,
                calculate_derivatives=calculate_derivatives)
        # Setting up calc.model.vector()
        self.calc.model.fit(images,
                            self.calc.descriptor,
                            self.calc.log,
                            self.calc.cores,
                            only_setup=True,
                            )
        # Truning off ConvergenceOccured exception and log_losses
        initial_raise_ConvergenceOccurred = \
            self.calc.model.lossfunction.raise_ConvergenceOccurred
        initial_log_losses = self.calc.model.lossfunction.log_losses
        self.calc.model.lossfunction.log_losses = False
        self.calc.model.lossfunction.raise_ConvergenceOccurred = False
        initial_state = self.calc.model.vector.copy()
        self.state = self.copy_state(initial_state)

        signal.signal(signal.SIGINT, self.set_user_exit)
        self.calc.log('\nAnnealing\n=========\n')
        bestState, bestLoss = self.anneal()
        # Taking the best state
        self.calc.model.vector = np.array(bestState)
        # Returning back the changed arguments
        self.calc.model.lossfunction.log_losses = initial_log_losses
        self.calc.model.lossfunction.raise_ConvergenceOccurred = \
            initial_raise_ConvergenceOccurred
        # cleaning up sessions
        self.calc.model.lossfunction._step = 0
        self.calc.model.lossfunction._cleanup()
        calc = self.calc

    @staticmethod
    def round_figures(x, n):
        """Returns x rounded to n significant figures."""
        return round(x, int(n - math.ceil(math.log10(abs(x)))))

    @staticmethod
    def time_string(seconds):
        """Returns time in seconds as a string formatted HHHH:MM:SS."""
        s = int(round(seconds))  # round to nearest second
        h, s = divmod(s, 3600)   # get hours and remainder
        m, s = divmod(s, 60)     # split remainder into minutes and seconds
        return '%4i:%02i:%02i' % (h, m, s)

    def save_state(self, fname=None):
        """Saves state"""
        if not fname:
            date = datetime.datetime.now().isoformat().split(".")[0]
            fname = date + "_loss_" + str(self.get_loss()) + ".state"
        print("Saving state to: %s" % fname)
        with open(fname, "w") as fh:
            pickle.dump(self.state, fh)

    def move(self, state):
        """Create a state change"""
        move_step = np.random.rand(len(state)) * 2. - 1.
        move_step *= 0.0005
        for _ in range(len(state)):
            state[_] = state[_] * (1 + move_step[_])
        return state

    def get_loss(self, state):
        """Calculate state's loss"""
        lossfxn = \
            self.calc.model.lossfunction.get_loss(np.array(state),
                                                  lossprime=False,)['loss']
        return lossfxn

    def set_user_exit(self, signum, frame):
        """Raises the user_exit flag, further iterations are stopped
        """
        self.user_exit = True

    def set_schedule(self, schedule):
        """Takes the output from `auto` and sets the attributes
        """
        self.Tmax = schedule['tmax']
        self.Tmin = schedule['tmin']
        self.steps = int(schedule['steps'])

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy : use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'copy':
            return state.copy()

    def update(self, step, T, L, acceptance, improvement):
        """Prints the current temperature, loss, acceptance rate,
        improvement rate, elapsed time, and remaining time.

        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the loss, moves that left the loss
        unchanged, and moves that increased the loss yet were reached by
        thermal excitation.

        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the loss.  At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the loss by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the loss
        are exhausted and moves that would increase the loss are no longer
        thermally accessible."""

        elapsed = time.time() - self.start
        if step == 0:
            self.calc.log('\n')
            header = ' %5s %12s %12s %7s %7s %10s %10s'
            self.calc.log(header % ('Step', 'Temperature', 'Loss (SSD)',
                                    'Accept', 'Improve', 'Elapsed',
                                    'Remaining'))
            self.calc.log(header % ('=' * 5, '=' * 12, '=' * 12,
                                    '=' * 7, '=' * 7, '=' * 10,
                                    '=' * 10,))
            self.calc.log(' %5i %12.2e %12.4e                   %s            '
                          % (step, T, L, self.time_string(elapsed)))
        else:
            remain = (self.steps - step) * (elapsed / step)
            self.calc.log(' %5i %12.2e %12.4e %7.2f%% %7.2f%% %s %s' %
                          (step, T, L, 100.0 * acceptance, 100.0 * improvement,
                           self.time_string(
                               elapsed), self.time_string(remain))),

    def anneal(self):
        """Minimizes the loss of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, loss): the best state and loss found.
        """
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        L = self.get_loss(self.state)
        prevState = self.copy_state(self.state)
        prevLoss = L
        bestState = self.copy_state(self.state)
        bestLoss = L
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, L, None, None)

        # Attempt moves to new states
        while step < (self.steps - 1) and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            self.state = self.move(self.state)
            L = self.get_loss(self.state)
            dL = L - prevLoss
            trials += 1
            if dL > 0.0 and math.exp(-dL / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                L = prevLoss
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dL < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevLoss = L
                if L < bestLoss:
                    bestState = self.copy_state(self.state)
                    bestLoss = L
            if self.updates > 1:
                if step // updateWavelength > (step - 1) // updateWavelength:
                    self.update(
                        step, T, L, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0

        # line break after progress output
        print('')

        self.state = self.copy_state(bestState)
        if self.save_state_on_exit:
            self.save_state()
        # Return best state and loss
        return bestState, bestLoss

    def auto(self, minutes, steps=2000):
        """Minimizes the loss of a system by simulated annealing with
        automatic selection of the temperature schedule.

        Keyword arguments:
        state -- an initial arrangement of the system
        minutes -- time to spend annealing (after exploring temperatures)
        steps -- number of steps to spend on each stage of exploration

        Returns the best state and loss found."""

        def run(T, steps):
            """Anneals a system at constant temperature and returns the state,
            loss, rate of acceptance, and rate of improvement."""
            L = self.get_loss()
            prevState = self.copy_state(self.state)
            prevLoss = L
            accepts, improves = 0, 0
            for step in range(steps):
                self.move()
                L = self.get_loss()
                dL = L - prevLoss
                if dL > 0.0 and math.exp(-dL / T) < random.random():
                    self.state = self.copy_state(prevState)
                    L = prevLoss
                else:
                    accepts += 1
                    if dL < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevLoss = L
            return L, float(accepts) / steps, float(improves) / steps

        step = 0
        self.start = time.time()

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = 0.0
        L = self.get_loss()
        self.update(step, T, L, None, None)
        while T == 0.0:
            step += 1
            self.move()
            T = abs(self.get_loss() - L)

        # Search for Tmax - a temperature that gives 98% acceptance
        L, acceptance, improvement = run(T, steps)

        step += steps
        while acceptance > 0.98:
            T = self.round_figures(T / 1.5, 2)
            L, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, L, acceptance, improvement)
        while acceptance < 0.98:
            T = self.round_figures(T * 1.5, 2)
            L, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, L, acceptance, improvement)
        Tmax = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > 0.0:
            T = self.round_figures(T / 1.5, 2)
            L, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, L, acceptance, improvement)
        Tmin = T

        # Calculate anneal duration
        elapsed = time.time() - self.start
        duration = self.round_figures(int(60.0 * minutes * step / elapsed), 2)

        print('')  # New line after auto() output
        # Don't perform anneal, just return params
        return {'tmax': Tmax, 'tmin': Tmin, 'steps': duration}


class MetaDict(dict):
    """Dictionary that can also store metadata. Useful for images dictionary
    so that images can still be iterated by keys."""
    metadata = {}
