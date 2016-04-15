import numpy as np

from ase.data import atomic_numbers
from ase.calculators.calculator import Parameters
from scipy.special import sph_harm
from ..utilities import Data, Logger
from .cutoffs import Cosine, Polynomial
from . import NeighborlistCalculator


class Zernike(object):

    """
    Class that calculates Zernike fingerprints.

    :param cutoff: Cutoff function. Can be also fed as a float representing the
                   radius above which neighbor interactions are ignored.
                   Default is 6.5 Angstroms.
    :type cutoff: object or float

    :param Gs: Dictionary of symbols and dictionaries for making symmetry
                functions. Either auto-genetrated, or given in the following
                form, for example:

               >>> Gs = {"Au": {"Au": 3., "O": 2.}, "O": {"Au": 5., "O": 10.}}

    :type Gs: dict

    :param nmax: Maximum degree of Zernike polynomials that will be included in
                 the fingerprint vector. Can be different values for different
                 species fed as a dictionary with chemical elements as keys.
    :type nmax: integer or dict

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

    def __init__(self, cutoff=Cosine(6.5), Gs=None, nmax=5, dblabel=None,
                 elements=None, version='2016.02', **kwargs):

        # Check of the version of descriptor, particularly if restarting.
        compatibleversions = ['2016.02', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use Zernike fingerprints'
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
                raise RuntimeError('This scheme only works '
                                   'in atom-centered mode. %s '
                                   'specified.' % mode)
        if len(kwargs) > 0:
            raise TypeError('Unexpected keyword arguments: %s' %
                            repr(kwargs))

        # If the cutoff is provided as a number, Cosine function will be used
        # by default.
        if isinstance(cutoff, int) or isinstance(cutoff, float):
            cutoff = Cosine(cutoff)

        # The parameters dictionary contains the minimum information
        # to produce a compatible descriptor; that is, one that gives
        # an identical fingerprint when fed an ASE image.
        p = self.parameters = Parameters(
            {'importname': '.descriptor.zernike.Zernike',
             'mode': 'atom-centered'})
        p.version = version
        p.cutoff = cutoff.Rc
        p.cutofffn = cutoff.__class__.__name__
        p.Gs = Gs
        p.nmax = nmax
        p.elements = elements

        self.dblabel = dblabel
        self.parent = None  # Can hold a reference to main Amp instance.

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to restart the calculator."""
        return self.parameters.tostring()

    def calculate_fingerprints(self, images, cores=1, fortran=False,
                               log=None, calculate_derivatives=False):
        """Calculates the fingerpints of the images, for the ones not already
        done."""

        if calculate_derivatives is True:
            import warnings
            warnings.warn('Zernike descriptor cannot train forces yet. '
                          'Force training automatically turnned off. ')
            calculate_derivatives = False

        log = Logger(file=None) if log is None else log

        if (self.dblabel is None) and hasattr(self.parent, 'dblabel'):
            self.dblabel = self.parent.dblabel
        self.dblabel = 'amp-data' if self.dblabel is None else self.dblabel

        p = self.parameters

        log('Cutoff radius: %.2f' % p.cutoff)
        log('Cutoff function: %s' % p.cutofffn)

        if p.elements is None:
            log('Finding unique set of elements in training data.')
            p.elements = set([atom.symbol for atoms in images.values()
                              for atom in atoms])
        p.elements = sorted(p.elements)
        log('%i unique elements included: ' % len(p.elements) +
            ', '.join(p.elements))

        log('Maximum degree of Zernike polynomials:')
        if isinstance(p.nmax, dict):
            for _ in p.nmax.keys():
                log(' %2s: %d' % (_, p.nmax[_]))
        else:
            log('nmax: %d' % p.nmax)

        if p.Gs is None:
            log('No coefficient for atomic density function supplied; '
                'creating defaults.')
            p.Gs = generate_coefficients(p.elements)
        log('Coefficients of atomic density function for each element:')
        for _ in p.Gs.keys():
            log(' %2s: %s' % (_, str(p.Gs[_])))

        # Counts the number of descriptors for each element.
        no_of_descriptors = {}
        for element in p.elements:
            count = 0
            if isinstance(p.nmax, dict):
                for n in xrange(p.nmax[element] + 1):
                    for l in xrange(n + 1):
                        if (n - l) % 2 == 0:
                            count += 1
            else:
                for n in xrange(p.nmax + 1):
                    for l in xrange(n + 1):
                        if (n - l) % 2 == 0:
                            count += 1
            no_of_descriptors[element] = count

        log('Number of descriptors for each element:')
        for element in p.elements:
            log(' %2s: %d' % (element, no_of_descriptors.pop(element)))

        log('Calculating neighborlists...', tic='nl')
        if not hasattr(self, 'neighborlist'):
            calc = NeighborlistCalculator(cutoff=p.cutoff)
            self.neighborlist = Data(filename='%s-neighborlists'
                                     % self.dblabel,
                                     calculator=calc)
        self.neighborlist.calculate_items(images, cores=cores, log=log)
        log('...neighborlists calculated.', toc='nl')

        log('Fingerprinting images...', tic='fp')
        if not hasattr(self, 'fingerprints'):
            calc = FingerprintCalculator(neighborlist=self.neighborlist,
                                         Gs=p.Gs,
                                         nmax=p.nmax,
                                         cutoff=p.cutoff,
                                         cutofffn=p.cutofffn)
            self.fingerprints = Data(filename='%s-fingerprints'
                                     % self.dblabel,
                                     calculator=calc)
        self.fingerprints.calculate_items(images, cores=cores, log=log)
        log('...fingerprints calculated.', toc='fp')


# Calculators #################################################################


class FingerprintCalculator:

    """For integration with .utilities.Data"""

    def __init__(self, neighborlist, Gs, nmax, cutoff, cutofffn):
        self.globals = Parameters({'cutoff': cutoff,
                                   'cutofffn': cutofffn,
                                   'Gs': Gs,
                                   'nmax': nmax})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprints'

        try:  # for scipy v <= 0.90
            from scipy import factorial as fac
        except ImportError:
            try:  # for scipy v >= 0.10
                from scipy.misc import factorial as fac
            except ImportError:  # for newer version of scipy
                from scipy.special import factorial as fac

        self.factorial = [fac(0.5 * _) for _ in xrange(2 * nmax + 2)]

    def calculate(self, image, key):
        """Makes a list of fingerprints, one per atom, for the fed image.
        """
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

        :param Rs: List of Cartesian atomic positions of neighbors.
        :type Rs: list of list of float

        :returns: list of float -- fingerprints for atom specified by its index
                                    and symbol.
        """

        home = self.atoms[index].position
        cutoff = self.globals.cutoff
        cutofffn = self.globals.cutofffn

        if cutofffn is 'Cosine':
            cutoff_fxn = Cosine(cutoff)
        elif cutofffn is 'Polynomial':
            cutoff_fxn = Polynomial(cutoff)

        fingerprint = []
        for n in xrange(self.globals.nmax + 1):
            for l in xrange(n + 1):
                if (n - l) % 2 == 0:
                    norm = 0.
                    for m in xrange(l + 1):
                        omega = 0.
                        for n_symbol, neighbor in zip(n_symbols, Rs):
                            x = (neighbor[0] - home[0]) / cutoff
                            y = (neighbor[1] - home[1]) / cutoff
                            z = (neighbor[2] - home[2]) / cutoff

                            rho = np.linalg.norm([x, y, z])

                            if rho > 0.:
                                theta = np.arccos(z / rho)
                            else:
                                theta = 0.

                            if x < 0.:
                                phi = np.pi + np.arctan(y / x)
                            elif 0. < x and y < 0.:
                                phi = 2 * np.pi + np.arctan(y / x)
                            elif 0. < x and 0. <= y:
                                phi = np.arctan(y / x)
                            elif x == 0. and 0. < y:
                                phi = 0.5 * np.pi
                            elif x == 0. and y < 0.:
                                phi = 1.5 * np.pi
                            else:
                                phi = 0.

                            ZZ = self.globals.Gs[symbol][n_symbol] * \
                                calculate_R(n, l, rho, self.factorial) * \
                                sph_harm(m, l, phi, theta) * \
                                cutoff_fxn(rho * cutoff)

                            # sum over neighbors
                            omega += np.conjugate(ZZ)
                        # sum over m values
                        if m == 0:
                            norm += omega * np.conjugate(omega)
                        else:
                            norm += 2. * omega * np.conjugate(omega)

                    fingerprint.append(norm.real)

        return symbol, fingerprint

# Auxiliary functions #########################################################


def binomial(n, k, factorial):
    """Returns C(n,k) = n!/(k!(n-k)!)."""

    assert n >= 0 and k >= 0 and n >= k, \
        'n and k should be non-negative integers with n >= k.'

    c = factorial[int(2 * n)] / \
        (factorial[int(2 * k)] * factorial[int(2 * (n - k))])
    return c


def calculate_R(n, l, rho, factorial):
    """
    Calculates R_{n}^{l}(rho) according to the last equation of wikipedia.
    """

    if (n - l) % 2 != 0:
        return 0
    else:
        value = 0.
        k = (n - l) / 2
        term1 = np.sqrt(2. * n + 3.)

        for s in xrange(k + 1):
            b1 = binomial(k, s, factorial)
            b2 = binomial(n - s - 1 + 1.5, k, factorial)
            value += ((-1) ** s) * b1 * b2 * (rho ** (n - 2. * s))

        value *= term1
        return value


def generate_coefficients(elements):
    """
    Automatically generates coefficients if not given by the user.

    :param elements: List of symbols of all atoms.
    :type elements: list of str

    :returns: dict of dicts
    """
    _G = {}
    for element in elements:
        _G[element] = atomic_numbers[element]
    G = {}
    for element in elements:
        G[element] = _G
    return G


if __name__ == "__main__":
    """Directly calling this module; apparently from another node.
    Calls should come as

    python -m amp.descriptor.example id hostname:port

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
        socket.send_pyobj(msg('<request>', 'cutofffn'))
        cutofffn = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'Gs'))
        Gs = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'nmax'))
        nmax = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()

        calc = FingerprintCalculator(neighborlist, Gs, nmax, cutoff, cutofffn)
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
