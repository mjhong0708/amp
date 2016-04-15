import numpy as np

from ase.data import atomic_numbers
from ase.calculators.calculator import Parameters
# should be imported as amp.utilities and not ..utilities, else readthedocs
# will nor read the docstring
from amp.utilities import Data, Logger
from .cutoffs import Cosine, Polynomial
from . import NeighborlistCalculator


class Gaussian(object):

    """
    Class that calculates Gaussian fingerprints (i.e., Behler-style).

    :param cutoff: Cutoff function. Can be also fed as a float representing the
                   radius above which neighbor interactions are ignored.
                   Default is 6.5 Angstroms.
    :type cutoff: object or float

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

    def __init__(self, cutoff=Cosine(6.5), Gs=None, dblabel=None,
                 elements=None, version=None, **kwargs):

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

        # If the cutoff is provided as a number, Cosine function will be used
        # by default.
        if isinstance(cutoff, int) or isinstance(cutoff, float):
            cutoff = Cosine(cutoff)

        # The parameters dictionary contains the minimum information
        # to produce a compatible descriptor; that is, one that gives
        # an identical fingerprint when fed an ASE image.
        p = self.parameters = Parameters(
            {'importname': '.descriptor.gaussian.Gaussian',
             'mode': 'atom-centered'})
        p.version = version
        p.cutoff = cutoff.Rc
        p.cutofffn = cutoff.__class__.__name__
        p.Gs = Gs
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
        done.  """
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
                                         cutoff=p.cutoff,
                                         cutofffn=p.cutofffn)
            self.fingerprints = Data(filename='%s-fingerprints'
                                     % self.dblabel,
                                     calculator=calc)
        self.fingerprints.calculate_items(images, cores=cores, log=log)
        log('...fingerprints calculated.', toc='fp')

        if calculate_derivatives:
            log('Calculating fingerprint derivatives of images...',
                tic='derfp')
            if not hasattr(self, 'fingerprintprimes'):
                calc = \
                    FingerprintPrimeCalculator(neighborlist=self.neighborlist,
                                               Gs=p.Gs,
                                               cutoff=p.cutoff,
                                               cutofffn=p.cutofffn)
                self.fingerprintprimes = \
                    Data(filename='%s-fingerprint-primes'
                         % self.dblabel,
                         calculator=calc)
            self.fingerprintprimes.calculate_items(
                images, cores=cores, log=log)
            log('...fingerprint derivatives calculated.', toc='derfp')


# Calculators #################################################################


class FingerprintCalculator:

    """For integration with .utilities.Data"""

    def __init__(self, neighborlist, Gs, cutoff, cutofffn):
        self.globals = Parameters({'cutoff': cutoff,
                                   'cutofffn': cutofffn,
                                   'Gs': Gs})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprints'

    def calculate(self, image, key):
        """Makes a list of fingerprints, one per atom, for the fed image."""
        self.atoms = image
        nl = self.keyed.neighborlist[key]
        fingerprints = []
        for atom in image:
            symbol = atom.symbol
            index = atom.index
            neighbors, offsets = nl[index]
            neighborsymbols = [image[_].symbol for _ in neighbors]
            Rs = [image.positions[neighbor] + np.dot(offset, image.cell)
                  for (neighbor, offset) in zip(neighbors, offsets)]
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

        :returns: list of float -- fingerprints for atom specified by its index
                                    and symbol.
        """
        home = self.atoms[index].position

        len_of_symmetries = len(self.globals.Gs[symbol])
        fingerprint = [None] * len_of_symmetries

        for count in xrange(len_of_symmetries):
            G = self.globals.Gs[symbol][count]

            if G['type'] == 'G2':
                ridge = calculate_G2(n_symbols, Rs, G['element'], G['eta'],
                                     self.globals.cutoff,
                                     self.globals.cutofffn, home)
            elif G['type'] == 'G4':
                ridge = calculate_G4(n_symbols, Rs, G['elements'], G['gamma'],
                                     G['zeta'], G['eta'], self.globals.cutoff,
                                     self.globals.cutofffn, home)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge

        return symbol, fingerprint


class FingerprintPrimeCalculator:

    """For integration with .utilities.Data"""

    def __init__(self, neighborlist, Gs, cutoff, cutofffn):
        self.globals = Parameters({'cutoff': cutoff,
                                   'cutofffn': cutofffn,
                                   'Gs': Gs})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprint_prime'

    def calculate(self, image, key):
        """Makes a list of fingerprint derivatives, one per atom,
        for the fed image."""
        self.atoms = image
        nl = self.keyed.neighborlist[key]
        fingerprintprimes = {}
        for atom in image:
            selfsymbol = atom.symbol
            selfindex = atom.index
            selfneighborindices, selfneighboroffsets = nl[selfindex]
            selfneighborsymbols = [
                image[_].symbol for _ in selfneighborindices]
            for i in xrange(3):
                # Calculating derivative of self atom fingerprints w.r.t.
                # coordinates of itself.
                nneighborindices, nneighboroffsets = nl[selfindex]
                nneighborsymbols = [image[_].symbol for _ in nneighborindices]

                Rs = [image.positions[_index] +
                      np.dot(_offset, image.get_cell())
                      for _index, _offset
                      in zip(nneighborindices,
                             nneighboroffsets)]

                der_indexfp = self.get_fingerprint_prime(
                    selfindex, selfsymbol,
                    nneighborindices,
                    nneighborsymbols,
                    Rs, selfindex, i)

                fingerprintprimes[
                    (selfindex, selfsymbol, selfindex, selfsymbol, i)] = \
                    der_indexfp
                # Calculating derivative of neighbor atom fingerprints w.r.t.
                # coordinates of self atom.
                for nindex, nsymbol, noffset in \
                        zip(selfneighborindices,
                            selfneighborsymbols,
                            selfneighboroffsets):
                    # for calculating forces, summation runs over neighbor
                    # atoms of type II (within the main cell only)
                    if noffset[0] == 0 and noffset[1] == 0 and noffset[2] == 0:
                        nneighborindices, nneighboroffsets = nl[nindex]
                        nneighborsymbols = \
                            [image[_].symbol for _ in nneighborindices]

                        Rs = [image.positions[_index] +
                              np.dot(_offset, image.get_cell())
                              for _index, _offset
                              in zip(nneighborindices,
                                     nneighboroffsets)]

                        # for calculating derivatives of fingerprints,
                        # summation runs over neighboring atoms of type
                        # I (either inside or outside the main cell)
                        der_indexfp = self.get_fingerprint_prime(
                            nindex, nsymbol,
                            nneighborindices,
                            nneighborsymbols,
                            Rs, selfindex, i)

                        fingerprintprimes[
                            (selfindex, selfsymbol, nindex, nsymbol, i)] = \
                            der_indexfp

        return fingerprintprimes

    def get_fingerprint_prime(self, index, symbol, n_indices, n_symbols, Rs,
                              m, i):
        """
        Returns the value of the derivative of G for atom with index and
        symbol with respect to coordinate x_{i} of atom index m. n_indices,
        n_symbols and Rs are lists of neighbors' indices, symbols and Cartesian
        positions, respectively.

        :param index: Index of the center atom.
        :type index: int
        :param symbol: Symbol of the center atom.
        :type symbol: str
        :param n_indices: List of neighbors' indices.
        :type n_indices: list of int
        :param n_symbols: List of neighbors' symbols.
        :type n_symbols: list of str
        :param Rs: List of Cartesian atomic positions.
        :type Rs: list of list of float
        :param m: Index of the pair atom.
        :type m: int
        :param i: Direction of the derivative; is an integer from 0 to 2.
        :type i: int

        :returns: list of float -- the value of the derivative of the
                                   fingerprints for atom with index and symbol
                                   with respect to coordinate x_{i} of atom
                                   index m.
        """

        len_of_symmetries = len(self.globals.Gs[symbol])
        Rindex = self.atoms.positions[index]
        der_fingerprint = [None] * len_of_symmetries

        for count in xrange(len_of_symmetries):
            G = self.globals.Gs[symbol][count]
            if G['type'] == 'G2':
                ridge = calculate_G2_prime(
                    n_indices,
                    n_symbols,
                    Rs,
                    G['element'],
                    G['eta'],
                    self.globals.cutoff,
                    self.globals.cutofffn,
                    index,
                    Rindex,
                    m,
                    i)
            elif G['type'] == 'G4':
                ridge = calculate_G4_prime(
                    n_indices,
                    n_symbols,
                    Rs,
                    G['elements'],
                    G['gamma'],
                    G['zeta'],
                    G['eta'],
                    self.globals.cutoff,
                    self.globals.cutofffn,
                    index,
                    Rindex,
                    m,
                    i)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])

            der_fingerprint[count] = ridge

        return der_fingerprint

# Auxiliary functions #########################################################


def calculate_G2(symbols, Rs, G_element, eta, cutoff, cutofffn, home,
                 fortran=False):
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
    :param cutofffn: Cutoff function that is used.
    :type cutofffn: str
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
                                          cutoff=cutoff, cutofffn=cutofffn,
                                          home=home)
    else:
        if cutofffn is 'Cosine':
            cutoff_fxn = Cosine(cutoff)
        elif cutofffn is 'Polynomial':
            cutoff_fxn = Polynomial(cutoff)
        ridge = 0.  # One aspect of a fingerprint :)
        len_of_symbols = len(symbols)
        for count in xrange(len_of_symbols):
            symbol = symbols[count]
            R = Rs[count]
            if symbol == G_element:
                Rij = np.linalg.norm(R - home)
                ridge += (np.exp(-eta * (Rij ** 2.) / (cutoff ** 2.)) *
                          cutoff_fxn(Rij))

    return ridge


def calculate_G4(symbols, Rs, G_elements, gamma, zeta, eta, cutoff, cutofffn,
                 home, fortran=False):
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
    :param cutofffn: Cutoff function that is used.
    :type cutofffn: str
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
                                         cutoff=cutoff, cutofffn=cutofffn,
                                         home=home)
    else:
        if cutofffn is 'Cosine':
            cutoff_fxn = Cosine(cutoff)
        elif cutofffn is 'Polynomial':
            cutoff_fxn = Polynomial(cutoff)
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
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                term *= cutoff_fxn(Rij)
                term *= cutoff_fxn(Rik)
                term *= cutoff_fxn(Rjk)
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


def Kronecker_delta(i, j):
    """
    Kronecker delta function.

    :param i: First index of Kronecker delta.
    :type i: int
    :param j: Second index of Kronecker delta.
    :type j: int

    :returns: int -- the value of the Kronecker delta.
    """
    if i == j:
        return 1.
    else:
        return 0.


def der_position_vector(a, b, m, i):
    """
    Returns the derivative of the position vector R_{ab} with respect to
        x_{i} of atomic index m.

    :param a: Index of the first atom.
    :type a: int
    :param b: Index of the second atom.
    :type b: int
    :param m: Index of the atom force is acting on.
    :type m: int
    :param i: Direction of force.
    :type i: int

    :returns: list of float -- the derivative of the position vector R_{ab}
                               with respect to x_{i} of atomic index m.
    """
    der_position_vector = [None, None, None]
    der_position_vector[0] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(0, i)
    der_position_vector[1] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(1, i)
    der_position_vector[2] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(2, i)

    return der_position_vector


def der_position(m, n, Rm, Rn, l, i):
    """
    Returns the derivative of the norm of position vector R_{mn} with
        respect to x_{i} of atomic index l.

    :param m: Index of the first atom.
    :type m: int
    :param n: Index of the second atom.
    :type n: int
    :param Rm: Position of the first atom.
    :type Rm: float
    :param Rn: Position of the second atom.
    :type Rn: float
    :param l: Index of the atom force is acting on.
    :type l: int
    :param i: Direction of force.
    :type i: int

    :returns: list of float -- the derivative of the norm of position vector
                               R_{mn} with respect to x_{i} of atomic index l.
    """
    Rmn = np.linalg.norm(Rm - Rn)
    # mm != nn is necessary for periodic systems
    if l == m and m != n:
        der_position = (Rm[i] - Rn[i]) / Rmn
    elif l == n and m != n:
        der_position = -(Rm[i] - Rn[i]) / Rmn
    else:
        der_position = 0.
    return der_position


def der_cos_theta(a, j, k, Ra, Rj, Rk, m, i):
    """
    Returns the derivative of Cos(theta_{ajk}) with respect to
        x_{i} of atomic index m.

    :param a: Index of the center atom.
    :type a: int
    :param j: Index of the first atom.
    :type j: int
    :param k: Index of the second atom.
    :type k: int
    :param Ra: Position of the center atom.
    :type Ra: float
    :param Rj: Position of the first atom.
    :type Rj: float
    :param Rk: Position of the second atom.
    :type Rk: float
    :param m: Index of the atom force is acting on.
    :type m: int
    :param i: Direction of force.
    :type i: int

    :returns: float -- derivative of Cos(theta_{ajk}) with respect to x_{i}
                       of atomic index m.
    """
    Raj_ = Ra - Rj
    Raj = np.linalg.norm(Raj_)
    Rak_ = Ra - Rk
    Rak = np.linalg.norm(Rak_)
    der_cos_theta = 1. / \
        (Raj * Rak) * np.dot(der_position_vector(a, j, m, i), Rak_)
    der_cos_theta += +1. / \
        (Raj * Rak) * np.dot(Raj_, der_position_vector(a, k, m, i))
    der_cos_theta += -1. / \
        ((Raj ** 2.) * Rak) * np.dot(Raj_, Rak_) * \
        der_position(a, j, Ra, Rj, m, i)
    der_cos_theta += -1. / \
        (Raj * (Rak ** 2.)) * np.dot(Raj_, Rak_) * \
        der_position(a, k, Ra, Rk, m, i)
    return der_cos_theta


def calculate_G2_prime(n_indices, symbols, Rs, G_element, eta, cutoff,
                       cutofffn, a, Ra, m, i, fortran=False):
    """
    Calculates coordinate derivative of G2 symmetry function for atom at
    index a and position Ra with respect to coordinate x_{i} of atom index
    m.

    :param n_indices: List of int of neighboring atoms.
    :type n_indices: list of int
    :param symbols: List of symbols of neighboring atoms.
    :type symbols: list of str
    :param Rs: List of Cartesian atomic positions of neighboring atoms.
    :type Rs: list of list of float
    :param G_element: Symmetry functions of the center atom.
    :type G_element: dict
    :param eta: Parameter of Behler symmetry functions.
    :type eta: float
    :param cutoff: Radius above which neighbor interactions are ignored.
    :type cutoff: float
    :param cutofffn: Cutoff function that is used.
    :type cutofffn: str
    :param a: Index of the center atom.
    :type a: int
    :param Ra: Position of the center atom.
    :type Ra: float
    :param m: Index of the atom force is acting on.
    :type m: int
    :param i: Direction of force.
    :type i: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :returns: float -- coordinate derivative of G2 symmetry function for atom
                       at index a and position Ra with respect to coordinate
                       x_{i} of atom index m.
    """
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_g2_prime(n_indices=list(n_indices),
                                                numbers=numbers, rs=Rs,
                                                g_number=G_number,
                                                g_eta=eta, cutoff=cutoff,
                                                cutofffn=cutofffn,
                                                aa=a, home=Ra, mm=m,
                                                ii=i)
    else:
        if cutofffn is 'Cosine':
            cutoff_fxn = Cosine(cutoff)
        elif cutofffn is 'Polynomial':
            cutoff_fxn = Polynomial(cutoff)
        ridge = 0.  # One aspect of a fingerprint :)
        len_of_symbols = len(symbols)
        for count in xrange(len_of_symbols):
            symbol = symbols[count]
            Rj = Rs[count]
            n_index = n_indices[count]
            if symbol == G_element:
                Raj = np.linalg.norm(Ra - Rj)
                term1 = (-2. * eta * Raj * cutoff_fxn(Raj) / (cutoff ** 2.) +
                         cutoff_fxn.prime(Raj))
                term2 = der_position(a, n_index, Ra, Rj, m, i)
                ridge += np.exp(- eta * (Raj ** 2.) / (cutoff ** 2.)) * \
                    term1 * term2
    return ridge


def calculate_G4_prime(n_indices, symbols, Rs, G_elements, gamma, zeta, eta,
                       cutoff, cutofffn, a, Ra, m, i, fortran=False):
    """
    Calculates coordinate derivative of G4 symmetry function for atom at
    index a and position Ra with respect to coordinate x_{i} of atom index m.

    :param n_indices: List of int of neighboring atoms.
    :type n_indices: list of int
    :param symbols: List of symbols of neighboring atoms.
    :type symbols: list of str
    :param Rs: List of Cartesian atomic positions of neighboring atoms.
    :type Rs: list of list of float
    :param G_elements: Symmetry functions of the center atom.
    :type G_elements: dict
    :param gamma: Parameter of Behler symmetry functions.
    :type gamma: float
    :param zeta: Parameter of Behler symmetry functions.
    :type zeta: float
    :param eta: Parameter of Behler symmetry functions.
    :type eta: float
    :param cutoff: Radius above which neighbor interactions are ignored.
    :type cutoff: float
    :param cutofffn: Cutoff function that is used.
    :type cutofffn: str
    :param a: Index of the center atom.
    :type a: int
    :param Ra: Position of the center atom.
    :type Ra: float
    :param m: Index of the atom force is acting on.
    :type m: int
    :param i: Direction of force.
    :type i: int
    :param fortran: If True, will use the fortran subroutines, else will not.
    :type fortran: bool

    :returns: float -- coordinate derivative of G4 symmetry function for atom
                       at index a and position Ra with respect to coordinate
                       x_{i} of atom index m.
    """
    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_g4_prime(n_indices=list(n_indices),
                                                numbers=numbers, rs=Rs,
                                                g_numbers=G_numbers,
                                                g_gamma=gamma,
                                                g_zeta=zeta, g_eta=eta,
                                                cutoff=cutoff,
                                                cutofffn=cutofffn,
                                                aa=a,
                                                home=Ra, mm=m,
                                                ii=i)
    else:
        if cutofffn is 'Cosine':
            cutoff_fxn = Cosine(cutoff)
        elif cutofffn is 'Polynomial':
            cutoff_fxn = Polynomial(cutoff)
        ridge = 0.
        counts = range(len(symbols))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([symbols[j], symbols[k]])
                if els != G_elements:
                    continue
                Rj = Rs[j]
                Rk = Rs[k]
                Raj_ = Rs[j] - Ra
                Raj = np.linalg.norm(Raj_)
                Rak_ = Rs[k] - Ra
                Rak = np.linalg.norm(Rak_)
                Rjk_ = Rs[j] - Rs[k]
                Rjk = np.linalg.norm(Rjk_)
                cos_theta_ajk = np.dot(Raj_, Rak_) / Raj / Rak
                c1 = (1. + gamma * cos_theta_ajk)
                c2 = cutoff_fxn(Raj)
                c3 = cutoff_fxn(Rak)
                c4 = cutoff_fxn(Rjk)
                if zeta == 1:
                    term1 = \
                        np.exp(- eta * (Raj ** 2. + Rak ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                else:
                    term1 = c1 ** (zeta - 1.) * \
                        np.exp(- eta * (Raj ** 2. + Rak ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                term2 = c2 * c3 * c4
                term3 = der_cos_theta(a, n_indices[j], n_indices[k], Ra, Rj,
                                      Rk, m, i)
                term4 = gamma * zeta * term3
                term5 = der_position(a, n_indices[j], Ra, Rj, m, i)
                term4 += -2. * c1 * eta * Raj * term5 / (cutoff ** 2.)
                term6 = der_position(a, n_indices[k], Ra, Rk, m, i)
                term4 += -2. * c1 * eta * Rak * term6 / (cutoff ** 2.)
                term7 = der_position(n_indices[j], n_indices[k], Rj, Rk, m, i)
                term4 += -2. * c1 * eta * Rjk * term7 / (cutoff ** 2.)
                term2 = term2 * term4
                term8 = cutoff_fxn.prime(Raj) * c3 * c4 * term5
                term9 = c2 * cutoff_fxn.prime(Rak) * c4 * term6
                term10 = c2 * c3 * cutoff_fxn.prime(Rjk) * term7

                term11 = term2 + c1 * (term8 + term9 + term10)
                term = term1 * term11
                ridge += term
        ridge *= 2. ** (1. - zeta)

    return ridge


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
        socket.send_pyobj(msg('<request>', 'cutofffn'))
        cutofffn = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'Gs'))
        Gs = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()

        calc = FingerprintCalculator(neighborlist, Gs, cutoff, cutofffn)
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

    elif purpose == 'calculate_fingerprint_primes':
        # Request variables.
        socket.send_pyobj(msg('<request>', 'cutoff'))
        cutoff = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'cutofffn'))
        cutofffn = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'Gs'))
        Gs = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()

        calc = FingerprintPrimeCalculator(neighborlist, Gs, cutoff,
                                          cutofffn)
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
