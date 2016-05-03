import numpy as np
from numpy import sqrt

from ase.data import atomic_numbers
from ase.calculators.calculator import Parameters
from scipy.special import sph_harm
from ..utilities import Data, Logger
from .cutoffs import Cosine, Polynomial
from . import NeighborlistCalculator
try:
    from .. import fmodules
except ImportError:
    fmodules = None


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

    :param fortran: If True, will use fortran modules, if False, will not.
    :type fortran: bool

    :raises: RuntimeError, TypeError
    """

    def __init__(self, cutoff=Cosine(6.5), Gs=None, nmax=5, dblabel=None,
                 elements=None, version='2016.02', mode='atom-centered',
                 fortran=True):

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

        # Check that the mode is atom-centered.
        if mode != 'atom-centered':
            raise RuntimeError('Zernike scheme only works '
                               'in atom-centered mode. %s '
                               'specified.' % mode)

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
        self.fortran = fortran
        self.parent = None  # Can hold a reference to main Amp instance.

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to restart the calculator."""
        return self.parameters.tostring()

    def calculate_fingerprints(self, images, cores=1, fortran=None,
                               log=None, calculate_derivatives=False):
        """Calculates the fingerpints of the images, for the ones not already
        done."""
        if fortran is None:
            fortran = self.fortran
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
                                         cutofffn=p.cutofffn,
                                         fortran=fortran)
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
                                               nmax=p.nmax,
                                               cutoff=p.cutoff,
                                               cutofffn=p.cutofffn,
                                               fortran=fortran)
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

    def __init__(self, neighborlist, Gs, nmax, cutoff, cutofffn, fortran):
        self.globals = Parameters({'cutoff': cutoff,
                                   'cutofffn': cutofffn,
                                   'Gs': Gs,
                                   'nmax': nmax})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprints'
        self.fortran = fortran

        try:  # for scipy v <= 0.90
            from scipy import factorial as fac
        except ImportError:
            try:  # for scipy v >= 0.10
                from scipy.misc import factorial as fac
            except ImportError:  # for newer version of scipy
                from scipy.special import factorial as fac

        self.factorial = [fac(0.5 * _) for _ in xrange(4 * nmax + 3)]

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
                        c_nlm = 0.
                        for n_symbol, neighbor in zip(n_symbols, Rs):
                            x = (neighbor[0] - home[0]) / cutoff
                            y = (neighbor[1] - home[1]) / cutoff
                            z = (neighbor[2] - home[2]) / cutoff
                            rho = np.linalg.norm([x, y, z])

                            if self.fortran:
                                Z_nlm = fmodules.calculate_z(n=n, l=l, m=m,
                                                             x=x, y=y, z=z,
                                                             factorial=self.factorial,
                                                             length=len(self.factorial))
                                Z_nlm = self.globals.Gs[symbol][n_symbol] * \
                                    Z_nlm * cutoff_fxn(rho * cutoff)

                            else:
                                # Alternative ways to calculate Z_nlm
#                                Z_nlm = self.globals.Gs[symbol][n_symbol] * \
#                                    calculate_Z(n, l, m, x, y, z, self.factorial) * \
#                                    cutoff_fxn(rho * cutoff)
#                                Z_nlm = self.globals.Gs[symbol][n_symbol] * \
#                                    calculate_Z2(n, l, m, x, y, z) * \
#                                    cutoff_fxn(rho * cutoff)

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

                                Z_nlm = self.globals.Gs[symbol][n_symbol] * \
                                    calculate_R(n, l, rho, self.factorial) * \
                                    sph_harm(m, l, phi, theta) * \
                                    cutoff_fxn(rho * cutoff)

                            # sum over neighbors
                            c_nlm += np.conjugate(Z_nlm)
                        # sum over m values
                        if m == 0:
                            norm += c_nlm * np.conjugate(c_nlm)
                        else:
                            norm += 2. * c_nlm * np.conjugate(c_nlm)

                    fingerprint.append(norm.real)

        return symbol, fingerprint


class FingerprintPrimeCalculator:

    """For integration with .utilities.Data"""

    def __init__(self, neighborlist, Gs, nmax, cutoff, cutofffn, fortran):
        self.globals = Parameters({'cutoff': cutoff,
                                   'cutofffn': cutofffn,
                                   'Gs': Gs,
                                   'nmax': nmax})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprint_prime'
        self.fortran = fortran

        try:  # for scipy v <= 0.90
            from scipy import factorial as fac
        except ImportError:
            try:  # for scipy v >= 0.10
                from scipy.misc import factorial as fac
            except ImportError:  # for newer version of scipy
                from scipy.special import factorial as fac

        self.factorial = [fac(0.5 * _) for _ in xrange(4 * nmax + 3)]

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
                              p, q):
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
        :param p: Index of the pair atom.
        :type p: int
        :param q: Direction of the derivative; is an integer from 0 to 2.
        :type q: int

        :returns: list of float -- the value of the derivative of the
                                   fingerprints for atom with index and symbol
                                   with respect to coordinate x_{i} of atom
                                   index m.
        """
        home = self.atoms[index].position
        cutoff = self.globals.cutoff
        # Cutofffn should be also added.
        cutofffn = self.globals.cutofffn

        if cutofffn is 'Cosine':
            cutoff_fxn = Cosine(cutoff)
        elif cutofffn is 'Polynomial':
            cutoff_fxn = Polynomial(cutoff)

        fingerprint_prime = []
        for n in xrange(self.globals.nmax + 1):
            for l in xrange(n + 1):
                if (n - l) % 2 == 0:
                    if self.fortran:  # fortran version; faster
                        G_numbers = [self.globals.Gs[symbol][elm]
                                     for elm in n_symbols]
                        numbers = [atomic_numbers[elm] for elm in n_symbols]
                        if len(Rs) == 0:
                            norm_prime = 0.
                        else:
                            norm_prime = \
                                fmodules.calculate_zernike_prime(n=n,
                                                                 l=l,
                                                                 n_length=len(
                                                                     n_indices),
                                                                 n_indices=list(
                                                                     n_indices),
                                                                 numbers=numbers,
                                                                 rs=Rs,
                                                                 g_numbers=G_numbers,
                                                                 cutoff=cutoff,
                                                                 indexx=index,
                                                                 home=home,
                                                                 p=p,
                                                                 q=q,
                                                                 fac_length=len(
                                                                     self.factorial),
                                                                 factorial=self.factorial)
                    else:
                        norm_prime = 0.
                        for m in xrange(l + 1):
                            c_nlm = 0.
                            c_nlm_prime = 0.
                            for n_index, n_symbol, neighbor in zip(n_indices,
                                                                   n_symbols,
                                                                   Rs):
                                x = (neighbor[0] - home[0]) / cutoff
                                y = (neighbor[1] - home[1]) / cutoff
                                z = (neighbor[2] - home[2]) / cutoff
                                rho = np.linalg.norm([x, y, z])

                                _Z_nlm = calculate_Z(n, l, m,
                                                     x, y, z,
                                                     self.factorial)
                                # Calculates Z_nlm
                                Z_nlm = _Z_nlm * \
                                    cutoff_fxn(rho * cutoff)

                                # Calculates Z_nlm_prime
                                Z_nlm_prime = _Z_nlm * \
                                    cutoff_fxn.prime(rho * cutoff) * \
                                    der_position(
                                        index, n_index, home, neighbor, p, q)

                                _Z_nlm_prime = calculate_Z_prime(n, l, m,
                                                                 x, y, z, q,
                                                                 self.factorial)

                                if (Kronecker(n_index, p) -
                                   Kronecker(index, p)) == 1:
                                    Z_nlm_prime += \
                                        cutoff_fxn(rho * cutoff) * \
                                        _Z_nlm_prime / cutoff
                                elif (Kronecker(n_index, p) -
                                      Kronecker(index, p)) == -1:
                                    Z_nlm_prime -= \
                                        cutoff_fxn(rho * cutoff) * \
                                        _Z_nlm_prime / cutoff

                                # sum over neighbors
                                c_nlm += self.globals.Gs[symbol][
                                    n_symbol] * np.conjugate(Z_nlm)
                                c_nlm_prime += self.globals.Gs[symbol][
                                    n_symbol] * np.conjugate(Z_nlm_prime)

                            # sum over m values
                            if m == 0:
                                norm_prime += 2. * c_nlm * \
                                    np.conjugate(c_nlm_prime)
                            else:
                                norm_prime += 4. * c_nlm * \
                                    np.conjugate(c_nlm_prime)

                    fingerprint_prime.append(norm_prime.real)

        return fingerprint_prime


# Auxiliary functions #########################################################


def binomial(n, k, factorial):
    """
    Returns C(n,k) = n!/(k!(n-k)!).
    """
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


def Kronecker(i, j):
    """
    Kronecker delta function.

    :param i: First index of Kronecker delta.
    :type i: int
    :param j: Second index of Kronecker delta.
    :type j: int

    :returns: int -- the value of the Kronecker delta.
    """
    if i == j:
        return 1
    else:
        return 0


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


def calculate_q(nu, k, l, factorial):
    """
    Calculates q_{kl}^{nu} according to the unnumbered equation afer Eq. (7) of
    "3D Zernike Descriptors for Content Based Shape Retrieval", Computer-Aided
    Design 36 (2004) 1047-1062.
    """
    result = ((-1) ** (k + nu)) * sqrt((2. * l + 4. * k + 3.) / 3.) * \
        binomial(k, nu, factorial) * \
        binomial(2. * k, k, factorial) * \
        binomial(2. * (k + l + nu) + 1., 2. * k, factorial) / \
        binomial(k + l + nu, k, factorial) / (2. ** (2. * k))

    return result


def calculate_Z(n, l, m, x, y, z, factorial):
    """
    Calculates Z_{nl}^{m}(x, y, z) according to the unnumbered equation afer
    Eq. (11) of "3D Zernike Descriptors for Content Based Shape Retrieval",
    Computer-Aided Design 36 (2004) 1047-1062.
    """
    value = 0.
    term1 = sqrt((2. * l + 1.) * factorial[int(2 * (l + m))] *
                 factorial[int(2 * (l - m))]) / factorial[int(2 * l)]
    term2 = 2. ** (-m)

    k = int((n - l) / 2.)
    for nu in xrange(k + 1):
        q = calculate_q(nu, k, l, factorial)
        for alpha in xrange(nu + 1):
            b1 = binomial(nu, alpha, factorial)
            for beta in xrange(nu - alpha + 1):
                b2 = binomial(nu - alpha, beta, factorial)
                term3 = q * b1 * b2
                for u in xrange(m + 1):
                    b5 = binomial(m, u, factorial)
                    term4 = ((-1.)**(m - u)) * b5 * (1j**u)
                    for mu in xrange(int((l - m) / 2.) + 1):
                        b6 = binomial(l, mu, factorial)
                        b7 = binomial(l - mu, m + mu, factorial)
                        term5 = ((-1.)**mu) * (2.**(-2. * mu)) * b6 * b7
                        for eta in xrange(mu + 1):
                            r = 2. * (eta + alpha) + u
                            s = 2. * (mu - eta + beta) + m - u
                            t = 2. * (nu - alpha - beta - mu) + l - m
                            value += term3 * term4 * term5 * \
                                binomial(mu, eta, factorial) * \
                                (x ** r) * (y ** s) * (z ** t)
    term6 = (1j) ** m
    value = term1 * term2 * term6 * value
    value = value / sqrt(4. * np.pi / 3.)

    return value


def calculate_Z_prime(n, l, m, x, y, z, p, factorial):
    """
    Calculates dZ_{nl}^{m}(x, y, z)/dR_{p} according to the unnumbered equation
    afer Eq. (11) of "3D Zernike Descriptors for Content Based Shape
    Retrieval", Computer-Aided Design 36 (2004) 1047-1062.
    """
    value = 0.
    term1 = sqrt((2. * l + 1.) * factorial[int(2 * (l + m))] *
                 factorial[int(2 * (l - m))]) / factorial[int(2 * l)]
    term2 = 2. ** (-m)

    k = int((n - l) / 2.)
    for nu in xrange(k + 1):
        q = calculate_q(nu, k, l, factorial)
        for alpha in xrange(nu + 1):
            b1 = binomial(nu, alpha, factorial)
            for beta in xrange(nu - alpha + 1):
                b2 = binomial(nu - alpha, beta, factorial)
                term3 = q * b1 * b2
                for u in xrange(m + 1):
                    term4 = ((-1.)**(m - u)) * binomial(
                        m, u, factorial) * (1j**u)
                    for mu in xrange(int((l - m) / 2.) + 1):
                        term5 = ((-1.)**mu) * (2.**(-2. * mu)) * \
                            binomial(l, mu, factorial) * \
                            binomial(l - mu, m + mu, factorial)
                        for eta in xrange(mu + 1):
                            r = 2 * (eta + alpha) + u
                            s = 2 * (mu - eta + beta) + m - u
                            t = 2 * (nu - alpha - beta - mu) + l - m
                            coefficient = term3 * term4 * \
                                term5 * binomial(mu, eta, factorial)
                            if p == 0:
                                if r != 0:
                                    value += coefficient * r * \
                                        (x ** (r - 1)) * (y ** s) * (z ** t)
                            elif p == 1:
                                if s != 0:
                                    value += coefficient * s * \
                                        (x ** r) * (y ** (s - 1)) * (z ** t)
                            elif p == 2:
                                if t != 0:
                                    value += coefficient * t * \
                                        (x ** r) * (y ** s) * (z ** (t - 1))
    term6 = (1j) ** m
    value = term1 * term2 * term6 * value
    value = value / sqrt(4. * np.pi / 3.)

    return value


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

    elif purpose == 'calculate_fingerprint_primes':
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

        calc = FingerprintPrimeCalculator(neighborlist, Gs, cutoff, cutofffn)
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
