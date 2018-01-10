import numpy as np

from ase.data import atomic_numbers
from ase.calculators.calculator import Parameters
from ..utilities import Data, Logger, importer
from .cutoffs import Cosine, dict2cutoff
NeighborList = importer('NeighborList')
try:
    from .. import fmodules
except ImportError:
    fmodules = None


class Botu(object):
    """Class that calculates Botu fingerprints (i.e., Behler-style).

    Parameters
    ----------
    cutoff : object or float
        Cutoff function, typically from amp.descriptor.cutoffs.  Can be also
        fed as a float representing the radius above which neighbor
        interactions are ignored; in this case a cosine cutoff function will be
        employed.  Default is a 6.5-Angstrom cosine cutoff.
    Gs : dict
        Dictionary of symbols and lists of dictionaries for making symmetry
        functions. Either auto-genetrated, or given in the following form, for
        example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

    dblabel : str
        Optional separate prefix/location for database files, including
        fingerprints, fingerprint derivatives, and neighborlists. This file
        location can be shared between calculator instances to avoid
        re-calculating redundant information. If not supplied, just uses the
        value from label.
    elements : list
        List of allowed elements present in the system. If not provided, will
        be found automatically.
    version : str
        Version of fingerprints.
    fortran : bool
        If True, will use fortran modules, if False, will not.
    mode : str
        Can be either 'atom-centered' or 'image-centered'.

    Raises
    ------
        RuntimeError
    """

    def __init__(self, cutoff=Cosine(6.5), Gs=None, dblabel=None,
                 elements=None, version=None, fortran=False, angular=False,
                 mode='atom-centered'):

        # Check of the version of descriptor, particularly if restarting.
        compatibleversions = ['2015.12', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use Botu fingerprints'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               ' newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        # Check that the mode is atom-centered.
        if mode != 'atom-centered':
            raise RuntimeError('Botu scheme only works '
                               'in atom-centered mode. %s '
                               'specified.' % mode)

        # If the cutoff is provided as a number, Cosine function will be used
        # by default.
        if isinstance(cutoff, int) or isinstance(cutoff, float):
            cutoff = Cosine(cutoff)
        # If the cutoff is provided as a dictionary, assume we need to load it
        # with dict2cutoff.
        if type(cutoff) is dict:
            cutoff = dict2cutoff(cutoff)

        # The parameters dictionary contains the minimum information
        # to produce a compatible descriptor; that is, one that gives
        # an identical fingerprint when fed an ASE image.
        p = self.parameters = Parameters(
            {'importname': '.descriptor.botu.Botu',
             'mode': 'atom-centered'})
        p.version = version
        p.cutoff = cutoff.todict()
        p.Gs = Gs
        p.elements = elements
        p.angular = self.angular = angular

        self.dblabel = dblabel
        self.fortran = fortran
        self.parent = None  # Can hold a reference to main Amp instance.

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to restart the calculator.
        """
        return self.parameters.tostring()

    def calculate_fingerprints(self, images, parallel=None, log=None,
                               calculate_derivatives=False):
        """Calculates the fingerpints of the images, for the ones not already
        done.

        Parameters
        ----------
        images : dict
            Dictionary of images; the key is a unique ID assigned to each
            image and each value is an ASE atoms object. Typically created
            from amp.utilities.hash_images.
        parallel : dict
            Configuration for parallelization. Should be in same form as in
            amp.Amp.
        log : Logger object
            Write function at which to log data. Note this must be a callable
            function.
        calculate_derivatives : bool
            Decides whether or not fingerprintprimes should also be calculated.
        """
        if parallel is None:
            parallel = {'cores': 1}
        log = Logger(file=None) if log is None else log

        if (self.dblabel is None) and hasattr(self.parent, 'dblabel'):
            self.dblabel = self.parent.dblabel
        self.dblabel = 'amp-data' if self.dblabel is None else self.dblabel

        p = self.parameters

        log('Cutoff function: %s' % repr(dict2cutoff(p.cutoff)))

        if p.elements is None:
            log('Finding unique set of elements in training data.')
            p.elements = set([atom.symbol for atoms in images.values()
                              for atom in atoms])
        p.elements = sorted(p.elements)
        log('%i unique elements included: ' % len(p.elements) +
            ', '.join(p.elements))

        if p.Gs is None:
            log('No symmetry functions supplied; creating defaults.')
            p.Gs = make_default_symmetry_functions(p.elements, self.angular)
        log('Number of symmetry functions for each element:')
        for _ in p.Gs.keys():
            log(' %2s: %i' % (_, len(p.Gs[_])))
        for element, fingerprints in p.Gs.items():
            log('{} feature vector functions:'.format(element))
            for index, fp in enumerate(fingerprints):
                if fp['type'] == 'G2':
                    log(' {}: {}, {}, eta = {}'
                        .format(index, fp['type'], fp['element'], fp['eta']))
                elif fp['type'] == 'G4':
                    log(' {}: {}, ({}, {}), eta={}, gamma={}, zeta={}'
                        .format(index, fp['type'], fp['elements'][0],
                                fp['elements'][1], fp['eta'], fp['gamma'],
                                fp['zeta']))
                else:
                    log(str(fp))

        log('Calculating neighborlists...', tic='nl')
        if not hasattr(self, 'neighborlist'):
            calc = NeighborlistCalculator(cutoff=p.cutoff['kwargs']['Rc'])
            self.neighborlist = \
                Data(filename='%s-neighborlists' % self.dblabel,
                     calculator=calc)
        self.neighborlist.calculate_items(images, parallel=parallel, log=log)
        log('...neighborlists calculated.', toc='nl')

        log('Fingerprinting images...', tic='fp')
        if not hasattr(self, 'fingerprints'):
            calc = FingerprintCalculator(neighborlist=self.neighborlist,
                                         Gs=p.Gs,
                                         cutoff=p.cutoff,
                                         fortran=self.fortran,
                                         angular=self.angular)
            self.fingerprints = Data(filename='%s-fingerprints'
                                     % self.dblabel,
                                     calculator=calc)
        self.fingerprints.calculate_items(images, parallel=parallel, log=log)
        log('...fingerprints calculated.', toc='fp')

        if calculate_derivatives:
            log('Calculating fingerprint derivatives...',
                tic='derfp')
            if not hasattr(self, 'fingerprintprimes'):
                calc = \
                    FingerprintPrimeCalculator(neighborlist=self.neighborlist,
                                               Gs=p.Gs,
                                               cutoff=p.cutoff,
                                               fortran=self.fortran,
                                               angular=self.angular)
                self.fingerprintprimes = \
                    Data(filename='%s-fingerprint-primes'
                         % self.dblabel,
                         calculator=calc)
            self.fingerprintprimes.calculate_items(
                images, parallel=parallel, log=log)
            log('...fingerprint derivatives calculated.', toc='derfp')


# Calculators #################################################################


# Neighborlist Calculator
class NeighborlistCalculator:
    """For integration with .utilities.Data

    For each image fed to calculate, a list of neighbors with offset distances
    is returned.

    Parameters
    ----------
    cutoff : float
        Radius above which neighbor interactions are ignored.
    """
    def __init__(self, cutoff):
        self.globals = Parameters({'cutoff': cutoff})
        self.keyed = Parameters()
        self.parallel_command = 'calculate_neighborlists'

    def calculate(self, image, key):
        """For integration with .utilities.Data

        For each image fed to calculate, a list of neighbors with offset
        distances is returned.

        Parameters
        ----------
        image : object
            ASE atoms object.
        key : str
            key of the image after being hashed.
        """
        cutoff = self.globals.cutoff
        n = NeighborList(cutoffs=[cutoff / 2.] * len(image),
                         self_interaction=False,
                         bothways=True,
                         skin=0.)
        n.update(image)
        return [n.get_neighbors(index) for index in range(len(image))]


class FingerprintCalculator:
    """For integration with .utilities.Data

    Parameters
    ----------
    neighborlist : list of str
        List of neighbors.
    Gs : dict
        Dictionary of symbols and lists of dictionaries for making symmetry
        functions. Either auto-genetrated, or given in the following form, for
        example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

    cutoff : float
        Radius above which neighbor interactions are ignored.
    fortran : bool
        If True, will use fortran modules, if False, will not.
    """
    def __init__(self, neighborlist, Gs, cutoff, fortran, angular):
        self.globals = Parameters({'cutoff': cutoff,
                                   'Gs': Gs,
                                   'angular': angular})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprints'
        self.fortran = fortran
        self.angular = angular

    def calculate(self, image, key):
        """Makes a list of fingerprints, one per atom, for the fed image.

        Parameters
        ----------
        image : object
            ASE atoms object.
        key : str
            key of the image after being hashed.
        """
        self.atoms = image
        nl = self.keyed.neighborlist[key]
        fingerprints = []
        for atom in image:
            symbol = atom.symbol
            index = atom.index
            neighborindices, neighboroffsets = nl[index]
            neighborsymbols = [image[_].symbol for _ in neighborindices]
            neighborpositions = \
                [image.positions[neighbor] + np.dot(offset, image.cell)
                 for (neighbor, offset) in zip(neighborindices,
                                               neighboroffsets)]
            indexfp = self.get_fingerprint(
                index, symbol, neighborsymbols, neighborpositions)
            fingerprints.append(indexfp)

        return fingerprints

    def get_fingerprint(self, index, symbol,
                        neighborsymbols, neighborpositions):
        """Returns the fingerprint of symmetry function values for atom
        specified by its index and symbol.

        neighborsymbols and neighborpositions are lists of neighbors' symbols
        and Cartesian positions, respectively.

        Parameters
        ----------
        index : int
            Index of the center atom.
        symbol : str
            Symbol of the center atom.
        neighborsymbols : list of str
            List of neighbors' symbols.
        neighborpositions : list of list of float
            List of Cartesian atomic positions.

        Returns
        -------
        symbol, fingerprint : list of float
            fingerprints for atom specified by its index and symbol.
        """
        Ri = self.atoms[index].position

        num_symmetries = len(self.globals.Gs[symbol])
        fingerprint = [None] * num_symmetries

        for count in range(num_symmetries):
            G = self.globals.Gs[symbol][count]

            if G['type'] == 'G2':
                ridge = calculate_G2(neighborsymbols, neighborpositions,
                                     G['element'], G['eta'],
                                     self.globals.cutoff, Ri, self.fortran)
            elif G['type'] == 'G4' and self.angular is True:
                ridge = calculate_G4(neighborsymbols, neighborpositions,
                                     G['elements'], G['gamma'],
                                     G['zeta'], G['eta'], self.globals.cutoff,
                                     Ri, self.fortran)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge

        return symbol, fingerprint


class FingerprintPrimeCalculator:
    """For integration with .utilities.Data

    Parameters
    ----------
    neighborlist : list of str
        List of neighbors.
    Gs : dict
        Dictionary of symbols and lists of dictionaries for making symmetry
        functions. Either auto-genetrated, or given in the following form, for
        example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

    cutoff : float
        Radius above which neighbor interactions are ignored.
    fortran : bool
        If True, will use fortran modules, if False, will not.
    """

    def __init__(self, neighborlist, Gs, cutoff, fortran, angular):
        self.globals = Parameters({'cutoff': cutoff,
                                   'Gs': Gs,
                                   'angular': angular})
        self.keyed = Parameters({'neighborlist': neighborlist})
        self.parallel_command = 'calculate_fingerprint_primes'
        self.fortran = fortran
        self.angular = angular

    def calculate(self, image, key):
        """Makes a list of fingerprint derivatives, one per atom,
        for the fed image.

        Parameters
        ----------
        image : object
            ASE atoms object.
        key : str
            key of the image after being hashed.
        """
        self.atoms = image
        nl = self.keyed.neighborlist[key]
        fingerprintprimes = {}
        for atom in image:
            symbol = atom.symbol
            index = atom.index
            neighborindices, neighboroffsets = nl[index]
            neighborsymbols = [image[_].symbol for _ in neighborindices]
            neighborpositions = \
                [image.positions[neighbor] + np.dot(offset, image.cell)
                 for (neighbor, offset) in zip(neighborindices,
                                               neighboroffsets)]
            for component in range(3):
                indexfp = self.get_fingerprintprime(
                          index, symbol, neighborsymbols, neighborpositions,
                          component)

                fingerprintprimes[index, symbol, component] = indexfp

        return fingerprintprimes

    def get_fingerprintprime(self, index, symbol, neighborsymbols,
                             neighborpositions, component):
        """Returns the fingerprint of symmetry function values for atom
        specified by its index and symbol.

        neighborsymbols and neighborpositions are lists of neighbors' symbols
        and Cartesian positions, respectively.

        Parameters
        ----------
        index : int
            Index of the center atom.
        symbol : str
            Symbol of the center atom.
        neighborsymbols : list of str
            List of neighbors' symbols.
        neighborpositions : list of list of float
            List of Cartesian atomic positions.

        Returns
        -------
        symbol, fingerprint : list of float
            fingerprints for atom specified by its index and symbol.
        """
        Ri = self.atoms[index].position

        num_symmetries = len(self.globals.Gs[symbol])
        fingerprintprime = [None] * num_symmetries

        for count in range(num_symmetries):
            G = self.globals.Gs[symbol][count]

            if G['type'] == 'G2':
                ridge = calculate_G2_prime(neighborsymbols, neighborpositions,
                                           G['element'], G['eta'],
                                           self.globals.cutoff, Ri,
                                           self.fortran, component)
            elif G['type'] == 'G4' and self.angular is True:
                ridge = calculate_G4_prime(neighborsymbols, neighborpositions,
                                           G['elements'], G['gamma'],
                                           G['zeta'], G['eta'],
                                           self.globals.cutoff, Ri,
                                           self.fortran)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])

            fingerprintprime[count] = ridge

        return fingerprintprime

# Auxiliary functions #########################################################


def calculate_G2(neighborsymbols,
                 neighborpositions, G_element, eta, cutoff, Ri, fortran):
    """Calculate G2 symmetry function.

    Ideally this will not be used but will be a template for how to build the
    fortran version (and serves as a slow backup if the fortran one goes
    uncompiled).  See Eq. 13a of the supplementary information of Khorshidi,
    Peterson, CPC(2016).

    Parameters
    ----------
    neighborsymbols : list of str
        List of symbols of all neighbor atoms.
    neighborpositions : list of list of floats
        List of Cartesian atomic positions.
    G_element : str
        Chemical symbol of the center atom.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        G2 fingerprint.
    """
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        neighbornumbers = \
            [atomic_numbers[symbol] for symbol in neighborsymbols]
        if len(neighbornumbers) == 0:
            ridge = 0.
        else:

            cutofffn = cutoff['name']
            Rc = cutoff['kwargs']['Rc']

            args_calculate_g2 = dict(
                    neighbornumbers=neighbornumbers,
                    neighborpositions=neighborpositions,
                    g_number=G_number,
                    g_eta=eta,
                    rc=Rc,
                    cutofffn=cutofffn,
                    ri=Ri
                    )
            if cutofffn == 'Polynomial':
                args_calculate_g2['p_gamma'] = cutoff['kwargs']['gamma']

            ridge = fmodules.calculate_g2(**args_calculate_g2)

    else:
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.  # One aspect of a fingerprint :)
        num_neighbors = len(neighborpositions)   # number of neighboring atoms
        for count in range(num_neighbors):
            symbol = neighborsymbols[count]
            Rj = neighborpositions[count]
            if symbol == G_element:
                Rij = np.linalg.norm(Rj - Ri)
                args_cutoff_fxn = dict(Rij=Rij)
                if cutoff['name'] == 'Polynomial':
                    args_cutoff_fxn['gamma'] = cutoff['kwargs']['gamma']
                ridge += (np.exp(-eta * (Rij ** 2.) / (Rc ** 2.)) *
                          cutoff_fxn(**args_cutoff_fxn))
    return ridge


def calculate_G4(neighborsymbols, neighborpositions,
                 G_elements, gamma, zeta, eta, cutoff,
                 Ri, fortran):
    """Calculate G4 symmetry function.

    Ideally this will not be used but will be a template for how to build the
    fortran version (and serves as a slow backup if the fortran one goes
    uncompiled).  See Eq. 13c of the supplementary information of Khorshidi,
    Peterson, CPC(2016).

    Parameters
    ----------
    neighborsymbols : list of str
        List of symbols of neighboring atoms.
    neighborpositions : list of list of floats
        List of Cartesian atomic positions of neighboring atoms.
    G_elements : list of str
        A list of two members, each member is the chemical species of one of
        the neighboring atoms forming the triangle with the center atom.
    gamma : float
        Parameter of Gaussian symmetry functions.
    zeta : float
        Parameter of Gaussian symmetry functions.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        G4 fingerprint.
    """
    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        neighbornumbers = \
            [atomic_numbers[symbol] for symbol in neighborsymbols]
        if len(neighborpositions) == 0:
            return 0.
        else:
            cutofffn = cutoff['name']
            Rc = cutoff['kwargs']['Rc']

            args_calculate_g4 = dict(
                    neighbornumbers=neighbornumbers,
                    neighborpositions=neighborpositions,
                    g_numbers=G_numbers,
                    g_gamma=gamma,
                    g_zeta=zeta,
                    g_eta=eta,
                    rc=Rc,
                    cutofffn=cutofffn,
                    ri=Ri
                    )
            if cutofffn == 'Polynomial':
                args_calculate_g4['p_gamma'] = cutoff['kwargs']['gamma']

            ridge = fmodules.calculate_g4(**args_calculate_g4)
            return ridge
    else:
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.
        counts = range(len(neighborpositions))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([neighborsymbols[j], neighborsymbols[k]])
                if els != G_elements:
                    continue
                Rij_vector = neighborpositions[j] - Ri
                Rij = np.linalg.norm(Rij_vector)
                Rik_vector = neighborpositions[k] - Ri
                Rik = np.linalg.norm(Rik_vector)
                Rjk_vector = neighborpositions[k] - neighborpositions[j]
                Rjk = np.linalg.norm(Rjk_vector)
                cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
                term = (1. + gamma * cos_theta_ijk) ** zeta
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               (Rc ** 2.))
                _Rij = dict(Rij=Rij)
                _Rik = dict(Rij=Rik)
                _Rjk = dict(Rij=Rjk)
                if cutoff['name'] == 'Polynomial':
                    _Rij['gamma'] = cutoff['kwargs']['gamma']
                    _Rik['gamma'] = cutoff['kwargs']['gamma']
                    _Rjk['gamma'] = cutoff['kwargs']['gamma']
                term *= cutoff_fxn(**_Rij)
                term *= cutoff_fxn(**_Rik)
                term *= cutoff_fxn(**_Rjk)
                ridge += term
        ridge *= 2. ** (1. - zeta)
        return ridge


def make_symmetry_functions(elements, type, etas, zetas=None, gammas=None):
    """Helper function to create Gaussian symmetry functions.
    Returns a list of dictionaries with symmetry function parameters
    in the format expected by the Gaussian class.

    Parameters
    ----------
    elements : list of str
        List of element types. The first in the list is considered the
        central element for this fingerprint. #FIXME: Does that matter?
    type : str
        Either G2 or G4.
    etas : list of floats
        eta values to use in G2 or G4 fingerprints
    zetas : list of floats
        zeta values to use in G4 fingerprints
    gammas : list of floats
        gamma values to use in G4 fingerprints

    Returns
    -------
    G : list of dicts
        A list, each item in the list contains a dictionary of fingerprint
        parameters.
    """
    if type == 'G2':
        G = [{'type': 'G2', 'element': element, 'eta': eta}
             for eta in etas
             for element in elements]
        return G
    elif type == 'G4':
        G = []
        for eta in etas:
            for zeta in zetas:
                for gamma in gammas:
                    for i1, el1 in enumerate(elements):
                        for el2 in elements[i1:]:
                            els = sorted([el1, el2])
                            G.append({'type': 'G4',
                                      'elements': els,
                                      'eta': eta,
                                      'gamma': gamma,
                                      'zeta': zeta})
        return G
    raise NotImplementedError('Unknown type: {}.'.format(type))


def make_default_symmetry_functions(elements, angular):
    """Makes symmetry functions as in Nano Letters 14:2670, 2014.


    Parameters
    ----------
    elements : list of str
        List of the elements, as in: ["C", "O", "H", "Cu"].

    Returns
    -------
    G : dict of lists
        The generated symmetry function parameters.
    """
    G = {}
    for element0 in elements:

        # Radial symmetry functions.
        etas = [0.05, 4., 20., 80.]
        _G = [{'type': 'G2', 'element': element, 'eta': eta}
              for eta in etas
              for element in elements]

        if angular:
            # Angular symmetry functions.
            # By default Botu only uses G2 element functions.
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


def calculate_G2_prime(neighborsymbols, neighborpositions, G_element, eta,
                       cutoff, Ri, fortran, component):
    """Calculate G2 symmetry function.

    Ideally this will not be used but will be a template for how to build the
    fortran version (and serves as a slow backup if the fortran one goes
    uncompiled).  See Eq. 13a of the supplementary information of Khorshidi,
    Peterson, CPC(2016).

    Parameters
    ----------
    neighborsymbols : list of str
        List of symbols of all neighbor atoms.
    neighborpositions : list of list of floats
        List of Cartesian atomic positions.
    G_element : str
        Chemical symbol of the center atom.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    fortran : bool
        If True, will use the fortran subroutines, else will not.

    Returns
    -------
    ridge : float
        G2 fingerprint.
    """
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        neighbornumbers = \
            [atomic_numbers[symbol] for symbol in neighborsymbols]
        if len(neighbornumbers) == 0:
            ridge = 0.
        else:

            cutofffn = cutoff['name']
            Rc = cutoff['kwargs']['Rc']

            args_calculate_g2 = dict(
                    neighbornumbers=neighbornumbers,
                    neighborpositions=neighborpositions,
                    g_number=G_number,
                    g_eta=eta,
                    rc=Rc,
                    cutofffn=cutofffn,
                    ri=Ri
                    )
            if cutofffn == 'Polynomial':
                args_calculate_g2['p_gamma'] = cutoff['kwargs']['gamma']

            ridge = fmodules.calculate_g2(**args_calculate_g2)

    else:
        """This is the original version that we have in Amp
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.  # One aspect of a fingerprint :)
        num_neighbors = len(neighborpositions)   # number of neighboring atoms
        for count in range(num_neighbors):
            symbol = neighborsymbols[count]
            Rj = neighborpositions[count]
            if symbol == G_element:
                Rij = np.linalg.norm(Rj - Ri)
                args_cutoff_fxn = dict(Rij=Rij)
                if cutoff['name'] == 'Polynomial':
                    args_cutoff_fxn['gamma'] = cutoff['kwargs']['gamma']
                ridge += (np.exp(-eta * (Rij ** 2.) / (Rc ** 2.)) *
                          cutoff_fxn(**args_cutoff_fxn))
        """
        Rc = cutoff['kwargs']['Rc']
        cutoff_fxn = dict2cutoff(cutoff)
        ridge = 0.  # One aspect of a fingerprint :)
        num_neighbors = len(neighborpositions)   # number of neighboring atoms
        for count in range(num_neighbors):
            symbol = neighborsymbols[count]
            Rj = neighborpositions[count]
            if symbol == G_element:
                Rij_vector = Rj - Ri
                Rij = np.linalg.norm(Rij_vector)
                args_cutoff_fxn = dict(Rij=Rij)
                if cutoff['name'] == 'Polynomial':
                    args_cutoff_fxn['gamma'] = cutoff['kwargs']['gamma']
                proj = (Ri[component] - Rj[component]) / Rij
                ridge += proj * (np.exp(-(Rij / eta) ** 2.) *
                                 cutoff_fxn(**args_cutoff_fxn))
    return ridge


if __name__ == "__main__":
    """Directly calling this module; apparently from another node.

    Calls should come as

        python -m amp.descriptor.botu id hostname:port

    This session will then start a zmq session with that socket, labeling
    itself with id. Instructions on what to do will come from the socket.
    """
    import sys
    import tempfile
    import zmq
    from ..utilities import MessageDictionary

    fortran = False if fmodules is None else True
    hostsocket = sys.argv[-1]
    proc_id = sys.argv[-2]
    msg = MessageDictionary(proc_id)

    # Send standard lines to stdout signaling process started and where
    # error is directed. This should be caught by pxssh. (This could
    # alternatively be done by zmq, but this works.)
    print('<amp-connect>')  # Signal that program started.
    sys.stderr = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                             suffix='.stderr')
    print('Log and error written to %s<stderr>' % sys.stderr.name)
    sys.stderr.write('initiated\n')
    sys.stderr.flush()

    # Establish client session via zmq; find purpose.
    context = zmq.Context()
    sys.stderr.write('context started\n')
    sys.stderr.flush()
    socket = context.socket(zmq.REQ)
    sys.stderr.write('socket started\n')
    sys.stderr.flush()
    socket.connect('tcp://%s' % hostsocket)
    sys.stderr.write('connection made\n')
    sys.stderr.flush()
    socket.send_pyobj(msg('<purpose>'))
    sys.stderr.write('message sent\n')
    sys.stderr.flush()
    purpose = socket.recv_pyobj()
    sys.stderr.write('purpose received\n')
    sys.stderr.flush()
    sys.stderr.write('purpose: %s \n' % purpose)
    sys.stderr.flush()

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
        socket.send_pyobj(msg('<request>', 'angular'))
        angular = socket.recv_pyobj()

        calc = FingerprintCalculator(neighborlist, Gs, cutoff,
                                     fortran, angular)
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
        socket.send_pyobj(msg('<request>', 'Gs'))
        Gs = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'neighborlist'))
        neighborlist = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'images'))
        images = socket.recv_pyobj()
        socket.send_pyobj(msg('<request>', 'angular'))
        angular = socket.recv_pyobj()

        calc = FingerprintPrimeCalculator(neighborlist, Gs, cutoff,
                                          fortran, angular)
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
