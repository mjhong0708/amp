import os
import sys
import shutil
import numpy as np
from string import Template
import time
import json
from scipy.stats.mstats import mquantiles
import tarfile
import tempfile

import ase.io

from ..utilities import hash_images, Logger, now
from .. import Amp

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

calc_text = """
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork


calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(),
           dblabel='../amp-db')
"""

train_line = "calc.train(images=hashed_images)"

script = """#!/usr/bin/env python
${headerlines}

from amp.utilities import TrainingConvergenceError, hash_images
from ase.parallel import paropen
from amp.stats.bootstrap import hash_with_duplicates
import os

${calc_text}

ensemble_index = int(os.path.split(os.getcwd())[-1])
trainfile = '../training-images/%i.traj' % ensemble_index
hashed_images = hash_with_duplicates(trainfile)

converged = True
try:
    ${train_line}
except TrainingConvergenceError:
    converged = False

f = paropen('converged', 'w')
f.write(str(converged))
f.close()
"""


class BootStrap:
    """A bootstrap ensemble calculator which serves as a wrapper around and
    Amp calculator. Initiate with an amp.utilities.Logger instance as log.

    If an existing trained bootstrap calculator is available, it can be
    loaded by providing its filename to the load keyword.

    Note that the 'train' method is meant to be a job-submission and
    -management script;
    e.g., it will typically be run at the command line to both submit jobs
    and monitor their convergence.
    """

    def __init__(self, load=None, log=None):
        if log is None:
            log = Logger(sys.stdout)
        self.log = log
        log('=' * 70)
        log('Amp bootstrap initiated.')
        log('Date: %s' % now(with_utc=True))
        if load is None:
            return

        with open(load) as f:
            calctexts = json.load(f)
        self.ensemble = []
        for calctext in calctexts:
            f = StringIO(calctext)
            calc = Amp.load(file=f)
            calc.log = Logger(None)
            self.ensemble.append(calc)
        log('Loaded ensemble of %i calculators.' % len(self.ensemble))

    def train(self, images, n=50, calc_text=calc_text, headerlines='',
              start_command='python run.py', sleep=0.1,
              train_line=train_line, label='bootstrap', expired=3600.):
        """Trains a bootstrap ensemble of calculators.


        This is set up to enable the submision of each as a job through
        the local queuing system, but can also run in serial.
        On first call to this method, jobs are created/submitted.
        On subsequent calls, jobs are analyzed for convergence.
        If all are converged, an ensemble is created and the training
        directory is archived.

        Parameters
        ----------
        n: int
           size of ensemble (number of calculators to train)
        calc_text: str
           text that is used to initiate the Amp calculator.
           see the example in this module in calc_text; must produce
           a 'calc' object
        headerlines: str
           lines in the top of the python script that will be submitted
           this would typically contain comment lines for the batching
           system, such as '#SBATCH -n=8...'
        start_command: str
           command to start the job in the current queuing system,
           such as 'sbatch run.py' ('run.py' is the scriptname here)
           for serial operation use 'python run.py'
        sleep : float
           time (s) to sleep between job submissions
        train_line: str
           line to use to train each amp instance; usually the default is
           fine but user may want to use this to insert additional keywords
           such as train_forces=False
        label: string
           label to give final trained calculator
        expired: float
           When checking jobs, age (s) of log file at which to consider
           that the job is no longer running (timed out) and should be
           restarted.

        Returns
        -------
        results: dict
            A dictionary indicating the state of training. This dictionary
            always contains a key 'complete' key with value of True or
            False indicating if training is complete. If False, also
            provides statistics on number converged.
        """

        log = self.log
        log('Train called.')
        trainingpath = '-'.join((label, 'training'))
        if os.path.exists(trainingpath):
            log('Path exists. Checking for which jobs are finished.')
            results = self._manage_jobs(n, trainingpath, expired,
                                        start_command, sleep, label)
            return results

        log('Training set: ' + str(images))
        images = hash_images(images)
        log('%i images in training set after hashing.' % len(images))
        image_keys = images.keys()

        originalpath = os.getcwd()
        trajpath = os.path.join(trainingpath, 'training-images')
        os.mkdir(trainingpath)
        os.mkdir(trajpath)

        log('Creating bootstrapped training images in %s.' % trajpath)
        for index in range(n):
            log(' Choosing images for %i.' % index)
            chosen = bootstrap(image_keys)
            log(' Writing trajectory for %i.' % index)
            traj = ase.io.Trajectory(
                os.path.join(trajpath, '%i.traj' % index), 'w')
            for key in chosen:
                traj.write(images[key])

        log('Creating and submitting jobs.')
        os.chdir(trainingpath)
        template = Template(script)
        pwd = os.getcwd()

        for index in range(n):
            os.mkdir('%i' % index)
            os.chdir('%i' % index)
            with open('run.py', 'w') as f:
                f.write(template.substitute({'headerlines': headerlines,
                                             'calc_text': calc_text,
                                             'train_line': train_line}))
            os.system(start_command)
            time.sleep(sleep)
            os.chdir(pwd)
        os.chdir(originalpath)
        return {'complete': False,
                'n_converged': 0}

    def _manage_jobs(self, n, trainingpath, expired, start_command, sleep,
                     label):
        """Checks the running jobs to see which have finished, tries
        to restart any that are stuck, and creates a bundled trajectory
        when everything is finished."""
        def clean_and_restart():
            for _ in os.listdir(os.getcwd()):
                if _ != 'run.py':
                    if os.path.isdir(_):
                        shutil.rmtree(_)
                    else:
                        os.remove(_)
            os.system(start_command)
            time.sleep(sleep)
            log('  ---> restarted.')

        log = self.log
        n_unfinished = 0
        n_converged = 0
        n_unconverged = 0
        n_expired = 0
        n_notstarted = 0
        pwd = os.getcwd()
        os.chdir(trainingpath)
        fulltrainingpath = os.getcwd()
        for index in range(n):
            os.chdir('%i' % index)
            if not os.path.exists('converged'):
                if not os.path.exists('amp-log.txt'):
                    log('%i: Not started; no amp-log.txt file.' % index)
                    n_notstarted += 1
                else:
                    age = time.time() - os.path.getmtime('amp-log.txt')
                    log('{:d}: Still running? No converged file. Age: '
                        '{:.1f} hr'.format(index, age / 3600.))
                    if age > expired:
                        log(' Assumed expired. Cleaning up directory and '
                            'restarting.')
                        n_expired += 1
                        clean_and_restart()
                    else:
                        n_unfinished += 1
                os.chdir(fulltrainingpath)
                continue
            with open('converged') as f:
                converged = f.read()

            if converged == 'True':
                log('%i: Converged.' % index)
                n_converged += 1
            else:
                log('%i: Not converged. Cleaning up directory to '
                    'restart job.' % index)
                n_unconverged += 1
                clean_and_restart()
            os.chdir(fulltrainingpath)
        log('')
        log('Stats:')
        log('%10i converged' % n_converged)
        log('%10i not yet started' % n_notstarted)
        log('%10i apparently still running' % n_unfinished)
        log('%10i did not converge, restarted' % n_unconverged)
        log('%10i expired, restarted' % n_expired)
        log('=' * 10)
        log('%10i total' % n)
        log('\n')

        if n_converged < n:
            log('Not all runs converged; not creating bundled amp '
                'calculator.')
            os.chdir(pwd)
            return {'complete': False,
                    'n_converged': n_converged}

        log('Creating bundled amp calculator.')
        ensemble = []
        for index in range(n):
            os.chdir('%i' % index)
            with open('amp.amp') as f:
                text = f.read()
            ensemble.append(text)
            os.chdir(fulltrainingpath)
        os.chdir(pwd)
        with open('%s.ensemble' % label, 'w') as f:
            json.dump(ensemble, f)
            log('Saved in json format as "%s.ensemble".' % label)
        log('Converting training directory into tar archive...')
        archive_directory(trainingpath)
        log('...converted.')
        return {'complete': True}

    def get_potential_energy(self, atoms, output=(.5,)):
        """Returns the potential energy from the ensemble for the atoms
        object.

        By default only returns the median prediction (50th percentile)
        of the ensemble, such that it works like a normal ASE calculator.
        To get uncertainty information, use the output keyword with the
        following codes:

            <q>: (where <q> is a float) return the q quantile of the
            ensemble (where the quantile is a decimal, as in 0.5 for 50th
            percentile)

            e: return the whole ensemble prediction as a list

        Join the arguments with commas. For example, to return the median
        prediction plus a centered spread covering 90% of the ensemble
        prediction, use output=[.5, .05, .95].
        If the ensemble is requested, it must be the last argument, e.g.,
        output=[.5, .025, .97.5, 'e'].
        Note a list is typically returned, but if only one attribute is
        requested it returns it as a float, so that it's ASE-like.
        """
        energies = [calc.get_potential_energy(atoms) for calc in self.ensemble]
        if output[-1] == 'e':
            quantiles = output[:-1]
            return_ensemble = True
        else:
            quantiles = output
            return_ensemble = False
        for quantile in quantiles:
            if (quantile > 1.0) or (quantile < 0.0):
                raise RuntimeError('Quantiles must be between 0 and 1.')
        result = mquantiles(energies, prob=quantiles)
        result = list(result)
        if return_ensemble:
            result.append(energies)
        if len(result) == 1:
            result == result[0]
        return result

    def get_forces(self, atoms, output=(.5,)):
        """Returns the atomic forces from the ensemble for the atoms
        object.

        By default only returns the median prediction (50th percentile)
        of the ensemble, such that it works like a normal ASE calculator.
        To get uncertainty information, use the output keyword with the
        following codes:

            <q>: (where <q> is a float) return the q quantile of the
            ensemble (where the quantile is a decimal, as in 0.5 for 50th
            percentile)

            e: return the whole ensemble prediction as a list

        Join the arguments with commas. For example, to return the median
        prediction plus a centered spread covering 90% of the ensemble
        prediction, use output=[.5, .05, .95].
        If the ensemble is requested, it must be the last argument, e.g.,
        output=[.5, .025, .97.5, 'e'].
        Note a list is typically returned, but if only one attribute is
        requested it returns it as a float, so that it's ASE-like.
        """
        forces = [calc.get_forces(atoms) for calc in self.ensemble]
        forces = np.array(forces)
        if output[-1] == 'e':
            quantiles = output[:-1]
            return_ensemble = True
        else:
            quantiles = output
            return_ensemble = False
        for quantile in quantiles:
            if (quantile > 1.0) or (quantile < 0.0):
                raise RuntimeError('Quantiles must be between 0 and 1.')
        # FIXME/ap: Had to switch to np.percentile from scipy mquantiles.
        # Because mquantiles doesn't support higher dimensions.
        # Should probably switch to percentiles throughout the code as
        # it's easier to read.
        percentiles = np.array(quantiles) * 100.
        result = np.percentile(forces, percentiles, axis=0)
        result = list(result)
        if return_ensemble:
            result.append(forces)
        if len(result) == 1:
            result == result[0]
        return result

    def get_atomic_energies(self, atoms, output=(.5,)):
        """ Returns the energy per atom from ensemble.
        The output parameter works as get_potential_energy."""
        if output[-1] == 'e':
            quantiles = output[:-1]
            return_ensemble = True
        else:
            quantiles = output
            return_ensemble = False
        for quantile in quantiles:
            if (quantile > 1.0) or (quantile < 0.0):
                raise RuntimeError('Percentiles must be between 0 and 1.')
        self.get_potential_energy(atoms)  # Assure calculation is fresh.
        atomic_energies = np.array([calc.model.atomic_energies for calc in
                                    self.ensemble])
        result = mquantiles(atomic_energies, prob=quantiles, axis=0)
        result = list(result)
        if return_ensemble:
            result.append(atomic_energies)
        if len(result) == 1:
            result == result[0]
        return result


def bootstrap(vector, size=None, return_missing=False):
    """Returns a randomly chosen, with replacement, version of the data
    set. If size is None returns a vector of same length.
    To pull from sample from multiple vectors, zip and unzip them like:

    >>> xsbs, ysbs = zip(*bootstrap(zip(xs, ys)))

    If return_missing == True, also finds and returns the missing elements
    not sampled from the vector as a second output.
    """

    size = len(vector) if size is None else size
    ids = np.random.choice(len(vector), size=size, replace=True)
    chosen = [vector[_] for _ in ids]
    if return_missing is False:
        return chosen
    unchosen = set(range(len(vector))).difference(set(ids))
    unchosen = [vector[_] for _ in unchosen]
    return chosen, unchosen


def hash_with_duplicates(images):
    """Creates new hash id's for duplicate images; new dictionary contains
    a redundant copy of each atoms object, so that the lossfunctions can be
    used as-is. Note will typically waste ~30% of the computational cost;
    it would be more efficient to update the calls inside the loss
    functions."""
    if not hasattr(images, 'keys'):
        images = hash_images(images)
    duplicates = images.metadata['duplicates']
    dict_images = dict(images)
    for oldhash, repititions in duplicates.items():
        for repitition in range(repititions - 1):
            newhash = '-'.join([oldhash, '%i' % (repitition + 1)])
            assert newhash not in dict_images
            dict_images[newhash] = images[oldhash]
    return dict_images


def archive_directory(source_dir):
    """Turns <source_dir> into a .tar.gz file and removes the original
    directory."""
    outputname = source_dir + '.tar.gz'
    if os.path.exists(outputname):
        raise RuntimeError('%s exists.' % outputname)
    with tarfile.open(outputname, 'w:gz') as tar:
        tar.add(source_dir)
    shutil.rmtree(source_dir)


class TrainingArchive:
    """Helper to get training trajectories and Amp calc instances from the
    training tar ball. Initialize with archive name. The get commands use
    the path the file would have had if the archive were extracted."""

    def __init__(self, name):
        self.tf = tarfile.open(name)

    def get_trajectory(self, path):
        # Doesn't work with extractfile because of numpy bug.
        tempdir = tempfile.mkdtemp()
        self.tf.extract(member=path, path=tempdir)
        return ase.io.Trajectory(os.path.join(tempdir, path))

    def get_amp_calc(self, path):
        return Amp.load(self.tf.extractfile(path))
