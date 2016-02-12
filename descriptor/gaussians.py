import numpy as np

from ase.data import atomic_numbers
from ase.calculators.neighborlist import NeighborList
from ase.calculators.calculator import Parameters

from ..utilities import Data, Logger


class Gaussians(object):

    """
    Class that calculates Gaussian fingerprints (i.e., Behler-style).

    :param cutoff: Radius above which neighbor interactions are ignored.
                   Default is 6.5 Angstroms.
    :type cutoff: float

    :param Gs: Dictionary of symbols and lists of dictionaries for making
               symmetry functions. Either auto-genetrated, or given in the
               following form, for example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

    :type Gs: dict

    :param dblabel: Optional separate prefix/location for database files,
                    including fingerprints, fingerprint derivatives, and
                    neighborlists. This file location can be shared between
                    calculator instances to avoid re-calculating redundant
                    information. If not supplied, just uses the value from
                    label.
    :type dblabel: str

    :param elements: List of allowed elements present in the system. If not
                     provided, will be found automatically.
    :type elements: list

    :param version: Version of fingerprints.
    :type version: str

    :raises: RuntimeError, TypeError
    """

    def __init__(self, cutoff=6.5, Gs=None, dblabel=None, elements=None,
                 version=None, **kwargs):

        # Check of the version of descriptor, particularly if restarting.
        compatibleversions = ['2015.12', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use Gaussian fingerprints'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               ' newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        # Check any extra kwargs fed.
        if 'mode' in kwargs:
            mode = kwargs.pop('mode')
            if mode != 'atom-centered':
                raise RuntimeError('Gaussian scheme only works '
                                   'in atom-centered mode. %s '
                                   'specified.' % mode)
        if len(kwargs) > 0:
            raise TypeError('Unexpected keyword arguments: %s' %
                            repr(kwargs))

        # The parameters dictionary contains the minimum information
        # to produce a compatible descriptor; that is, one that gives
        # an identical fingerprint when fed an ASE image.
        p = self.parameters = Parameters(
            {'importname': '.descriptor.gaussians.Gaussians',
             'mode': 'atom-centered'})
        p.version = version
        p.cutoff = cutoff
        p.Gs = Gs
        p.elements = elements

        self.dblabel = dblabel
        self.parent = None  # Can hold a reference to main Amp instance.

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to restart the calculator."""
        return self.parameters.tostring()

    def calculate_fingerprints(self, images, cores=1, fortran=False,
                               log=None):
        """Calculates the fingerpints of the images, for the ones not
        already done.  """
        log = Logger(file=None) if log is None else log

        if (self.dblabel is None) and hasattr(self.parent, 'dblabel'):
            self.dblabel = self.parent.dblabel
        self.dblabel = 'amp-data' if self.dblabel is None else self.dblabel

        p = self.parameters

        if p.elements is None:
            log('Finding unique set of elements in training data.')
            p.elements = set([atom.symbol for atoms in images.values()
                              for atom in atoms])
        p.elements = sorted(p.elements)
        log('%i unique elements included: ' % len(p.elements) +
            ', '.join(p.elements))

        if p.Gs is None:
            log('No symmetry functions supplied; creating defaults.')
            p.Gs = make_symmetry_functions(p.elements)
        log('Symmetry functions for each element:')
        for _ in p.Gs.keys():
            log(' %2s: %i' % (_, len(p.Gs[_])))

        log('Calculating neighborlists...', tic='nl')
        if not hasattr(self, 'neighborlist'):
            nl = Data(filename='%s-neighborlists' % self.dblabel,
                      calculator=NeighborlistCalculator(cutoff=p.cutoff))
            self.neighborlist = nl
        self.neighborlist.calculate_items(images, cores=cores, log=log)
        log('...neighborlists calculated.', toc='nl')

        log('Fingerprinting images...', tic='fp')
        if not hasattr(self, 'fingerprints'):
            calc = FingerprintCalculator(neighborlist=self.neighborlist,
                                         Gs=p.Gs,
                                         cutoff=p.cutoff)
            self.fingerprints = Data(filename='%s-fingerprints'
                                     % self.dblabel,
                                     calculator=calc)
        self.fingerprints.calculate_items(images, cores=cores, log=log)
        log('...fingerprints calculated.', toc='fp')


# Calculators #################################################################

class NeighborlistCalculator:

    """For integration with .utilities.Data
    For each image fed to calculate, a list of neighbors with offset
    distances is returned.
    """

    def __init__(self, cutoff):
        self.globals = Parameters({'cutoff': cutoff})
        self.keyed = Parameters()
        self.parallel_command = 'calculate_neighborlists'

    def calculate(self, image, key):
        cutoff = self.globals.cutoff
        n = NeighborList(cutoffs=[cutoff / 2.] * len(image),
                         self_interaction=False,
                         bothways=True,
                         skin=0.)
        n.update(image)
        return [n.get_neighbors(index) for index in xrange(len(image))]


class FingerprintCalculator:

    """For integration with .utilities.Data"""

    def __init__(self, neighborlist, Gs, cutoff):
        self.globals = Parameters({'cutoff': cutoff,
                                   'Gs': Gs})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprints'

    def calculate(self, image, key):
        """Makes a list of fingerprints, one per atom, for the fed image."""
        nl = self.keyed.neighborlist[key]
        fingerprints = []
        for atom in image:
            symbol = atom.symbol
            index = atom.index
            neighbors, offsets = nl[index]
            neighborsymbols = [image[_].symbol for _ in neighbors]
            Rs = [image.positions[neighbor] + np.dot(offset, image.cell)
                  for (neighbor, offset) in zip(neighbors, offsets)]
            self.atoms = image
            indexfp = self.get_fingerprint(index, symbol, neighborsymbols, Rs)
            fingerprints.append(indexfp)

        return fingerprints

    def get_fingerprint(self, index, symbol, n_symbols, Rs):
        """
        Returns the fingerprint of symmetry function values for atom
        specified by its index and symbol. n_symbols and Rs are lists of
        neighbors' symbols and Cartesian positions, respectively.

        :param index: Index of the center atom.
        :type index: int

        :param symbol: Symbol of the center atom.
        :type symbol: str

        :param n_symbols: List of neighbors' symbols.
        :type n_symbols: list of str

        :param Rs: List of Cartesian atomic positions.
        :type Rs: list of list of float

        :returns: list of float -- the symmetry function values for atom
                                   specified by its index and symbol.
        """
        home = self.atoms[index].position

        len_of_symmetries = len(self.globals.Gs[symbol])
        fingerprint = [None] * len_of_symmetries
        count = 0
        while count < len_of_symmetries:
            G = self.globals.Gs[symbol][count]

            if G['type'] == 'G2':
                ridge = calculate_G2(n_symbols, Rs, G['element'], G['eta'],
                                     self.globals.cutoff, home)
            elif G['type'] == 'G4':
                ridge = calculate_G4(n_symbols, Rs, G['elements'], G['gamma'],
                                     G['zeta'], G['eta'], self.globals.cutoff,
                                     home)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge
            count += 1

        return symbol, fingerprint

# Auxiliary functions #########################################################


def calculate_G2(symbols, Rs, G_element, eta, cutoff, home, fortran=False):
    """
    Calculate G2 symmetry function. Ideally this will not be used but
    will be a template for how to build the fortran version (and serves as
    a slow backup if the fortran one goes uncompiled).

    :param symbols: List of symbols of all atoms.
    :type symbols: list of str
    :param Rs: List of Cartesian atomic positions.
    :type Rs: list of list of float
    :param G_element: Symmetry functions of the center atom.
    :type G_element: dict
    :param eta: Parameter of Gaussian symmetry functions.
    :type eta: float
    :param cutoff: Radius above which neighbor interactions are ignored.
    :type cutoff: float
    :param home: Index of the center atom.
    :type home: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :returns: float -- G2 fingerprint.
    """
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_g2(numbers=numbers, rs=Rs,
                                          g_number=G_number, g_eta=eta,
                                          cutoff=cutoff, home=home)
    else:
        ridge = 0.  # One aspect of a fingerprint :)
        len_of_symbols = len(symbols)
        count = 0
        while count < len_of_symbols:
            symbol = symbols[count]
            R = Rs[count]
            if symbol == G_element:
                Rij = np.linalg.norm(R - home)
                ridge += (np.exp(-eta * (Rij ** 2.) / (cutoff ** 2.)) *
                          cutoff_fxn(Rij, cutoff))
            count += 1

    return ridge


def calculate_G4(symbols, Rs, G_elements, gamma, zeta, eta, cutoff, home,
                 fortran=False):
    """
    Calculate G4 symmetry function. Ideally this will not be used but
    will be a template for how to build the fortran version (and serves as
    a slow backup if the fortran one goes uncompiled).

    :param symbols: List of symbols of neighboring atoms.
    :type symbols: list of str
    :param Rs: List of Cartesian atomic positions of neighboring atoms.
    :type Rs: list of list of float
    :param G_elements: Symmetry functions of the center atom.
    :type G_elements: dict
    :param gamma: Parameter of Gaussian symmetry functions.
    :type gamma: float
    :param zeta: Parameter of Gaussian symmetry functions.
    :type zeta: float
    :param eta: Parameter of Gaussian symmetry functions.
    :type eta: float
    :param cutoff: Radius above which neighbor interactions are ignored.
    :type cutoff: float
    :param home: Index of the center atom.
    :type home: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :returns: float -- G4 fingerprint.
    """

    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            return 0.
        else:
            return fmodules.calculate_g4(numbers=numbers, rs=Rs,
                                         g_numbers=G_numbers, g_gamma=gamma,
                                         g_zeta=zeta, g_eta=eta,
                                         cutoff=cutoff, home=home)
    ridge = 0.
    counts = range(len(symbols))
    for j in counts:
        for k in counts[(j + 1):]:
            els = sorted([symbols[j], symbols[k]])
            if els != G_elements:
                continue
            Rij_ = Rs[j] - home
            Rij = np.linalg.norm(Rij_)
            Rik_ = Rs[k] - home
            Rik = np.linalg.norm(Rik_)
            Rjk = np.linalg.norm(Rs[j] - Rs[k])
            cos_theta_ijk = np.dot(Rij_, Rik_) / Rij / Rik
            term = (1. + gamma * cos_theta_ijk) ** zeta
            term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.))
            term *= cutoff_fxn(Rij, cutoff)
            term *= cutoff_fxn(Rik, cutoff)
            term *= cutoff_fxn(Rjk, cutoff)
            ridge += term
    ridge *= 2. ** (1. - zeta)
    return ridge


def make_symmetry_functions(elements):
    """
    Makes symmetry functions as in Nano Letters function by Artrith.
    Elements is a list of the elements, as in ["C", "O", "H", "Cu"].
    G[0] = {"type":"G2", "element": "O", "eta": 0.0009}
    G[40] = {"type":"G4", "elements": ["O", "Au"], "eta": 0.0001,
    "gamma": 1.0, "zeta": 1.0}

    If G (a list) is fed in, this will add to it and return an expanded
    version. If not, it will create a new one.

    :param elements: List of symbols of all atoms.
    :type elements: list of str

    :returns: dict of lists -- symmetry functions if not given by the user.
    """
    G = {}
    for element0 in elements:

        # Radial symmetry functions.
        etas = [0.05, 4., 20., 80.]
        _G = [{'type': 'G2', 'element': element, 'eta': eta}
              for eta in etas
              for element in elements]

        # Angular symmetry functions.
        etas = [0.005]
        zetas = [1., 4.]
        gammas = [+1., -1.]
        for eta in etas:
            for zeta in zetas:
                for gamma in gammas:
                    for i1, el1 in enumerate(elements):
                        for el2 in elements[i1:]:
                            els = sorted([el1, el2])
                            _G.append({'type': 'G4',
                                       'elements': els,
                                       'eta': eta,
                                       'gamma': gamma,
                                       'zeta': zeta})
        G[element0] = _G
    return G


def cutoff_fxn(Rij, Rc):
    """
    Cosine cutoff function in Parinello-Behler method.

    :param Rc: Radius above which neighbor interactions are ignored.
    :type Rc: float
    :param Rij: Distance between pair atoms.
    :type Rij: float

    :returns: float -- the vaule of the cutoff function.
    """
    if Rij > Rc:
        return 0.
    else:
        return 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)


if __name__ == "__main__":
    """Directly calling this module; apparently from another node.
    Calls should come as

    python -m amp.descriptor.gaussian id hostname:port

    This session will then start a zmq session with that socket, labeling
    itself with id. Instructions on what to do will come from the socket.
    """
    import sys
    import tempfile
    import zmq
    from ..utilities import MessageDictionary

    hostsocket = sys.argv[-1]
    proc_id = sys.argv[-2]
    msg = MessageDictionary(proc_id)

    # Send standard lines to stdout signaling process started and where
    # error is directed. This should be caught by pxssh. (This could
    # alternatively be done by zmq, but this works.)
    print('<amp-connect>')  # Signal that program started.
    sys.stderr = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                             suffix='.stderr')
    print('stderr written to %s<stderr>' % sys.stderr.name)

    # Establish client session via zmq; find purpose.
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://%s' % hostsocket)
    socket.send_pyobj(msg('<purpose>'))
    purpose = socket.recv_pyobj()

    if purpose == 'calculate_neighborlists':
        # Request variables.
        socket.send_pyobj(msg('<request>', 'cutoff'))
        cutoff = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()
        # sys.stderr.write(str(images)) # Just to see if they are there.

        # Perform the calculations.
        calc = NeighborlistCalculator(cutoff=cutoff)
        neighborlist = {}
        # for key in images.iterkeys():
        while len(images) > 0:
            key, image = images.popitem()  # Reduce memory.
            neighborlist[key] = calc.calculate(image, key)

        # Send the results.
        socket.send_pyobj(msg('<result>', neighborlist))
        socket.recv_string()  # Needed to complete REQ/REP.

    elif purpose == 'calculate_fingerprints':
        # Request variables.
        socket.send_pyobj(msg('<request>', 'cutoff'))
        cutoff = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'Gs'))
        Gs = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()

        calc = FingerprintCalculator(neighborlist, Gs, cutoff)
        result = {}
        while len(images) > 0:
            key, image = images.popitem()  # Reduce memory.
            result[key] = calc.calculate(image, key)
            if len(images) % 100 == 0:
                socket.send_pyobj(msg('<info>', len(images)))
                socket.recv_string()  # Needed to complete REQ/REP.

        # Send the results.
        socket.send_pyobj(msg('<result>', result))
        socket.recv_string()  # Needed to complete REQ/REP.

    else:
        raise NotImplementedError('purpose %s unknown.' % purpose)
