import os
import sys
import shutil
import numpy as np
import tempfile
import platform
from getpass import getuser
from socket import gethostname

import ase
from ase.calculators.calculator import Calculator, Parameters
try:
    from ase import __version__ as aseversion
except ImportError:
    # We're on ASE 3.10 or older
    from ase.version import version as aseversion

from .utilities import make_filename
from .utilities import hash_images
from .utilities import Logger, string2dict, logo, now, assign_cores


class Amp(Calculator, object):

    """
    Atomistic Machine-Learning Potential (Amp) ASE calculator

    :param descriptor: Class representing local atomic environment.
    :type descriptor: object

    :param regression: Class representing the regression method. Can be only
                       NeuralNetwork for now. Input arguments for NeuralNetwork
                       are hiddenlayers, activation, weights, and scalings; for
                       more information see docstring for the class
                       NeuralNetwork.
    :type regression: object

    :param load: Path for loading an existing parameters of Amp calculator.
    :type load: str

    :param label: Default prefix/location used for all files.
    :type label: str

    :param dblabel: Optional separate prefix/location for database files,
                    including fingerprints, fingerprint derivatives, and
                    neighborlists. This file location can be shared between
                    calculator instances to avoid re-calculating redundant
                    information. If not supplied, just uses the value from
                    label.
    :type dblabel: str

    :param cores: Can specify cores to use for parallel training;
                  if None, will determine from environment
    :type cores: int

    :raises: RuntimeError
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, descriptor, model, label='amp', dblabel=None,
                 cores=None, **kwargs):

        Calculator.__init__(self, label=label, **kwargs)

        log = Logger(make_filename(self.label, '-log.txt'))
        self.log = log
        self._printheader(log)

        # Note the following are properties: these are setter functions.
        self.descriptor = descriptor
        self.model = model
        self.cores = cores  # Note this calls 'assign_cores'.

        self.dblabel = label if dblabel is None else dblabel

    @property
    def cores(self):
        return self._cores

    @cores.setter
    def cores(self, cores):
        self._cores = assign_cores(cores, log=self.log)

    @property
    def descriptor(self):
        return self._descriptor

    @descriptor.setter
    def descriptor(self, descriptor):
        descriptor.parent = self  # gives the descriptor object a reference to
        # the main Amp instance. Then descriptor can pull parameters directly
        # from Amp without needing them to be passed in each method call.
        self._descriptor = descriptor
        self.reset()  # Clears any old calculations.

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        model.parent = self  # gives the model object a reference to the main
        # Amp instance. Then model can pull parameters directly from Amp
        # without needing them to be passed in each method call.
        self._model = model
        self.reset()  # Clears any old calculations.

    @classmethod
    def load(Cls, filename, Descriptor=None, Model=None, **kwargs):
        """Attempts to load calculators and return a new instance of Amp.
        Only a filename is required, in typical cases.

        If using a home-rolled descriptor or model, also supply
        uninstantiated classes to those models, as in Model=MyModel.

        Any additional keyword arguments (such as fortran=True) can be
        fed through to Amp.
        """
        with open(filename) as f:
            text = f.read()

        # Unpack parameter dictionaries.
        p = string2dict(text)
        for key in ['descriptor', 'model']:
            p[key] = string2dict(p[key])

        # If modules are not specified, find them.
        if Descriptor is None:
            Descriptor = importhelper(p['descriptor'].pop('importname'))
        if Model is None:
            Model = importhelper(p['model'].pop('importname'))
        # Key 'importname' and the value removed so that it is not splatted
        # into the keyword arguments used to instantiate in the next line.

        # Instantiate the descriptor and model.
        descriptor = Descriptor(**p['descriptor'])
        # ** sends all the key-value pairs at once.
        model = Model(**p['model'])

        # Instantiate Amp.
        calc = Cls(descriptor=descriptor, model=model, **kwargs)
        calc.log('Loaded file: %s' % filename)
        return calc

    def set(self, **kwargs):
        """
        Function to set parameters. For now, this doesn't do anything
        as all parameters are within the model and descriptor.
        """
        changed_parameters = Calculator.set(self, **kwargs)
        if len(changed_parameters) > 0:
            self.reset()

    def set_label(self, label):
        """
        Sets label, ensuring that any needed directories are made.

        :param label: Default prefix/location used for all files.
        :type label: str
        """
        Calculator.set_label(self, label)

        # Create directories for output structure if needed.
        if self.label:
            if (self.directory != os.curdir and
                    not os.path.isdir(self.directory)):
                os.makedirs(self.directory)

    def calculate(self, atoms, properties, system_changes):
        """
        Calculation of the energy of system and forces of all atoms.
        """
        # The inherited method below just sets the atoms object,
        # if specified, to self.atoms.
        Calculator.calculate(self, atoms, properties, system_changes)

        log = self.log
        log('Amp calculator started.')

        images = hash_images([self.atoms])
        key = images.keys()[0]

        if properties == ['energy']:
            self.descriptor.calculate_fingerprints(images=images,
                                                   log=log,
                                                   calculate_derivatives=False)
            energy = self.model.get_energy(self.descriptor.fingerprints[key])
            self.results['energy'] = energy

        if properties == ['forces']:
            self.descriptor.calculate_fingerprints(images=images,
                                                   log=log,
                                                   calculate_derivatives=True)
            # for calculating forces, energies should be calculated first,
            # at least for neural networks
            energy = self.model.get_energy(self.descriptor.fingerprints[key])
            self.results['energy'] = energy
            forces = \
                self.model.get_forces(self.descriptor.fingerprints[key],self.descriptor.derfingerprints[key])
            self.results['forces'] = forces

    def train(self,
              images,
              overwrite=False,
              energy_tol=0.001,
              force_tol=0.005,
              force_coefficient=None,
              ):
        """
        Fits the model to the training images.

        :param images: List of ASE atoms objects with positions, symbols,
                       energies, and forces in ASE format. This is the training
                       set of data. This can also be the path to an ASE
                       trajectory (.traj) or database (.db) file. Energies can
                       be obtained from any reference, e.g. DFT calculations.
        :type images: list or str

        :param overwrite: If a trained output file with the same name exists,
                          overwrite it.
        :type overwrite: bool

        :param energy_tol: Threshold energy per atom rmse at which simulation
                            is converged.
        :type energy_tol: float

        :param force_tol: Threshold force rmse at which simulation is
                           converged. The default value is in unit of eV/Ang.
                           If 'force_tol = None', forces will not be trained.
        :type force_tol: float

        :param force_coefficient: Coefficient of the force contribution in the
                                  cost function.
        :type force_coefficient: float
        """

        log = self.log
        log('Amp training started. ' + now() + '\n')

        log('Hashing training images.')
        images = hash_images(images, log=log)

        log('Descriptor: %s' % self.descriptor.__class__.__name__)
        log('Model: %s' % self.model.__class__.__name__)

        log('\nDescriptor\n==========')
        calculate_derivatives = True if force_coefficient is not None else False
        self.descriptor.calculate_fingerprints(images=images,
                                               cores=self.cores,
                                               log=log,
                                               calculate_derivatives=calculate_derivatives)

        if force_tol is None:
            if not force_coefficient:
                force_coefficient = 0.
        elif not force_coefficient:
            force_coefficient = (energy_tol / force_tol)**2.

        energy_coefficient = 1.

        log('\nModel fitting\n=============')
        result = self.model.fit(trainingimages=images,
                                descriptor=self.descriptor,
                                energy_coefficient=energy_coefficient,
                                force_coefficient=force_coefficient,
                                log=log,
                                cores=self.cores,)

        if result is True:
            log('Amp successfully trained. Saving current parameters.')
            filename = self.label + '.amp'
        else:
            log('Amp not trained successfully. Saving current parameters.')
            filename = make_filename(self.label, '-untrained-parameters.amp')
        filename = self.save(filename, overwrite)
        log('Parameters saved in file "%s".' % filename)

    def save(self, filename, overwrite=False):
        """Saves the calculator in way that it can be re-opened with
        load."""
        if os.path.exists(filename):
            if overwrite is False:
                oldfilename = filename
                filename = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                       suffix='.amp').name
                self.log('File "%s" exists. Instead saving to "%s".' %
                         (oldfilename, filename))
            else:
                oldfilename = tempfile.NamedTemporaryFile(mode='w',
                                                          delete=False,
                                                          suffix='.amp').name

                self.log('Overwriting file: "%s". Moving original to "%s".'
                         % (filename, oldfilename))
                shutil.move(filename, oldfilename)
        descriptor = self.descriptor.tostring()
        model = self.model.tostring()
        p = Parameters({'descriptor': descriptor,
                        'model': model})
        p.write(filename)
        return filename

    def _printheader(self, log):
        """Prints header to log file; inspired by that in GPAW."""
        log(logo)
        log('Amp: Atomistic Machine-learning Package')
        log('Developed by Andrew Peterson, Alireza Khorshidi, and others,')
        log('Brown University.')
        log(' PI Website: http://brown.edu/go/catalyst')
        log(' Official repository: http://bitbucket.org/andrewpeterson/amp')
        log(' Official documentation: http://amp.readthedocs.org/')
        log('=' * 70)
        log('User: %s' % getuser())
        log('Hostname: %s' % gethostname())
        log('Date: %s' % now(with_utc=True))
        uname = platform.uname()
        log('Architecture: %s' % uname[4])
        log('PID: %s' % os.getpid())
        log('Version: %s' % 'NOT NUMBERED YET.')  # FIXME/ap. Look at GPAW
        log('Python: v{0}.{1}.{2}: %s'.format(*sys.version_info[:3]) %
            sys.executable)
        log('Amp: %s' % os.path.dirname(os.path.abspath(__file__)))
        log('ASE v%s: %s' % (aseversion, os.path.dirname(ase.__file__)))
        log('NumPy v%s: %s' %
            (np.version.version, os.path.dirname(np.__file__)))
        # SciPy is not a strict dependency.
        try:
            import scipy
            log('SciPy v%s: %s' %
                (scipy.version.version, os.path.dirname(scipy.__file__)))
        except ImportError:
            log('SciPy: not available')
        # ZMQ an pxssh are only necessary for parallel calculations.
        try:
            import zmq
            log('ZMQ/PyZMQ v%s/v%s: %s' %
                (zmq.zmq_version(), zmq.pyzmq_version(),
                 os.path.dirname(zmq.__file__)))
        except ImportError:
            log('ZMQ: not available')
        try:
            from pexpect import pxssh
            log('pxssh: %s' % os.path.dirname(pxssh.__file__))
        except ImportError:
            log('pxssh: Not available')
        log('=' * 70)


def importhelper(importname):
    """Manually compiled list of available modules. This is to prevent the
    execution of arbitrary (potentially malicious) code.

    However, since there is an `eval` statement in string2dict maybe this
    is silly.
    """
    if importname == '.descriptor.gaussian.Gaussian':
        from .descriptor.gaussian import Gaussian as Module
    elif importname == '.model.neuralnetwork.NeuralNetwork':
        from .model.neuralnetwork import NeuralNetwork as Module
    elif importname == '.model.LossFunction':
        from .model import LossFunction as Module
    else:
        raise NotImplementedError(
            'Attempt to import the module %s. Was this intended? '
            'If so, trying manually importing this module and '
            'feeding it to Amp.load. To avoid this error, this '
            'module can be added to amp.importhelper.' %
            importname)

    return Module
