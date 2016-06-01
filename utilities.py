#!/usr/bin/env python

import numpy as np
import hashlib
import time
import os
from ase import io as aseio
from ase.parallel import paropen
import shelve
from datetime import datetime


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
    sublist_lengths = [N//n if _ >= (N % n) else N//n + 1 for _ in range(n)]
    sublists = []
    for sublist_length in sublist_lengths:
        sublists.append([masterlist.pop() for _ in xrange(sublist_length)])
    return sublists


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
            from getpass import getuser
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
            log(' Establishing worker sessions.')
            processes = []
            for workerhostname, nprocesses in cores.iteritems():
                n = len(processes)
                processes.extend([establish_ssh(_ + n) for _ in
                                  range(nprocesses)])

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
            self.tic = time.time()

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
                tic = self.tic
            else:
                tic = self.tics[toc]
            dt = (time.time() - tic) / 60.
            dt = ' %.1f min.' % dt
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
        log(' ...hashing completed.', toc='hash')
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
