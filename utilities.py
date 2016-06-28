#!/usr/bin/env python

import numpy as np
import hashlib
import time
import os
from ase import io as aseio
from ase.parallel import paropen
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

    # FIXME/ap sqlitedict probably behaves the same, but supports
    # multi-thread access.

    # FIXME/ap even better may be mongodb, which is designed to hold
    # json-like objects and looks like it is gaining popularity

    def __init__(self, filename, db=shelve, calculator=None):
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
        if self.d:
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
    dictionary. If ordered is True, returns an OrderedDict.
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
                images = io.read(images)

        # images converted to dictionary form; key is hash of image.
        log('Hashing images...', tic='hash')
        dict_images = {}
        if ordered is True:
            from collections import OrderedDict
            dict_images = OrderedDict()
        for image in images:
            hash = hash_image(image)
            if hash in dict_images.keys():
                log('Warning: Duplicate image (based on identical hash).'
                    ' Was this expected? Hash: %s' % hash)
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
    calc.descriptor.calculate_fingerprints(images=images,
                                           cores=calc.cores,
                                           log=calc.log,
                                           calculate_derivatives=calculate_derivatives)

    vector = calc.model.vector.copy()

    # FIXME: AKh: Should read from filename, after it is saved.
    lossfunction = LossFunction(energy_coefficient=0.0,
                                force_coefficient=1.0,
                                cores=calc.cores,
                                )
    calc.model.lossfunction = lossfunction

    # Set up local loss function.
    lossfunction.attach_model(calc.model,
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
            ax.set_xlabel('parameter no %i' % count)
            ax.set_ylabel('loss function')
            pdf.savefig(fig)
            pyplot.close(fig)
            count += 1

    calc.log(' ...loss functions plotted.', toc='plot')
