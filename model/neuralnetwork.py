import numpy as np
from collections import OrderedDict

from ase.calculators.calculator import Parameters

from ..regression import Regressor
from . import LossFunction, calculate_fingerprints_range
from . import Model


class NeuralNetwork(Model):

    """
    Class that implements a basic feed-forward neural network.

    :param hiddenlayers: Dictionary of chemical element symbols and
                         architectures of their corresponding hidden layers of
                         the conventional neural network. Number of nodes of
                         last layer is always one corresponding to energy.
                         However, number of nodes of first layer is equal to
                         three times number of atoms in the system in the case
                         of no descriptor, and is equal to length of symmetry
                         functions of the descriptor. Can be fed as:

                         >>> hiddenlayers = (3, 2,)

                         for example, in which a neural network with two hidden
                         layers, the first one having three nodes and the
                         second one having two nodes is assigned (to the whole
                         atomic system in the no descriptor case, and to each
                         chemical element in the fingerprinting scheme).
                         In the fingerprinting scheme, neural network for each
                         species can be assigned seperately, as:

                         >>> hiddenlayers = {"O":(3,5), "Au":(5,6)}

                         for example.
    :type hiddenlayers: dict

    :param activation: Assigns the type of activation funtion. "linear" refers
                       to linear function, "tanh" refers to tanh function, and
                       "sigmoid" refers to sigmoid function.
    :type activation: str

    :param weights: In the case of no descriptor, keys correspond to layers
                    and values are two dimensional arrays of network weight.
                    In the fingerprinting scheme, keys correspond to chemical
                    elements and values are dictionaries with layer keys and
                    network weight two dimensional arrays as values. Arrays are
                    set up to connect node i in the previous layer with node j
                    in the current layer with indices w[i,j]. The last value
                    for index i corresponds to bias. If weights is not given,
                    arrays will be randomly generated.
    :type weights: dict

    :param scalings: In the case of no descriptor, keys are "intercept" and
                     "slope" and values are real numbers. In the fingerprinting
                     scheme, keys correspond to chemical elements and values
                     are dictionaries with "intercept" and "slope" keys and
                     real number values. If scalings is not given, it will be
                     randomly generated.
    :type scalings: dict

    .. note:: Dimensions of weight two dimensional arrays should be consistent
              with hiddenlayers.

    :raises: RuntimeError, NotImplementedError
    """
    ###########################################################################

    def __init__(self, hiddenlayers=(5, 5), activation='tanh', weights=None,
                 scalings=None, fprange=None, regressor=None, mode=None,
                 lossfunction=None, version=None):

        # Version check, particularly if restarting.
        compatibleversions = ['2015.12', ]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use NeuralNetwork'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               'newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        # The parameters dictionary contains the minimum information
        # to produce a compatible model; e.g., one that gives
        # the identical energy (and/or forces) when fed a fingerprint.
        p = self.parameters = Parameters()
        p.importname = '.model.neuralnetwork.NeuralNetwork'
        p.version = version
        p.hiddenlayers = hiddenlayers
        p.weights = weights
        p.scalings = scalings
        p.fprange = fprange
        p.activation = activation
        p.mode = mode

        # Checking that the activation function is given correctly:
        if activation not in ['linear', 'tanh', 'sigmoid']:
            _ = ('Unknown activation function %s; must be one of '
                 '"linear", "tanh", or "sigmoid".' % activation)
            raise NotImplementedError(_)

        self.regressor = regressor
        self.parent = None  # Can hold a reference to main Amp instance.
        self.lossfunction = lossfunction

        # Reset local variables corresponding to energy.
        self.o = {}
        self.D = {}
        self.delta = {}
        self.ohat = {}

    def fit(self, trainingimages, descriptor, energy_coefficient,
            force_coefficient, log, cores):
        """Fit the model parameters such that the fingerprints can be used to
        describe the energies in trainingimages. log is the logging object.
        descriptor is a descriptor object, as would be in calc.descriptor.
        """
        # Set all parameters and report to logfile.
        self.cores = cores

        if self.lossfunction is None:
            self.lossfunction = LossFunction(energy_coefficient,
                                             force_coefficient,
                                             cores=self.cores,)
        if self.regressor is None:
            self.regressor = Regressor()

        p = self.parameters
        tp = self.trainingparameters = Parameters()
        tp.trainingimages = trainingimages
        tp.descriptor = descriptor

        if p.mode is None:
            p.mode = descriptor.parameters.mode
        else:
            assert p.mode == descriptor.parameters.mode
        log(' Regression in %s mode.' % p.mode)

        if 'fprange' not in p or p.fprange is None:
            log('Calculating new fingerprint range.')
            p.fprange = calculate_fingerprints_range(descriptor,
                                                     trainingimages)

        if p.mode == 'atom-centered':
            # If hiddenlayers is a tuple/list, convert to a dictionary.
            if not hasattr(p.hiddenlayers, 'keys'):
                p.hiddenlayers = {element: p.hiddenlayers
                                  for element in p.fprange.keys()}

        log('Hidden-layer structure:')
        if p.mode == 'image-centered':
            log(' %s' % str(p.hiddenlayers))
        elif p.mode == 'atom-centered':
            for item in p.hiddenlayers.items():
                log(' %2s: %s' % item)

        if p.weights is None:
            log('Initializing with random weights.')
            if p.mode == 'image-centered':
                raise NotImplementedError('Needs to be coded.')
            elif p.mode == 'atom-centered':
                p.weights = get_random_weights(p.hiddenlayers, p.activation,
                                               None, p.fprange)
        else:
            log('Initial weights already present.')

        if p.scalings is None:
            log('Initializing with random scalings.')
            if p.mode == 'image-centered':
                raise NotImplementedError('Need to code.')
            elif p.mode == 'atom-centered':
                p.scalings = get_random_scalings(trainingimages, p.activation,
                                                 p.fprange.keys())
        else:
            log('Initial scalings already present.')

        # Regress the model.
        result = self.regressor.regress(model=self, log=log)
        return result  # True / False

    @property
    def vector(self):
        """Access to get or set the model parameters (weights, scaling for
        each network) as a single vector, useful in particular for
        regression."""
        if self.parameters['weights'] is None:
            return None
        if not hasattr(self, 'ravel'):
            p = self.parameters
            self.ravel = Raveler(p.weights, p.scalings)
        return self.ravel.to_vector(weights=p.weights, scalings=p.scalings)

    @vector.setter
    def vector(self, vector):
        if not hasattr(self, 'ravel'):
            p = self.parameters
            self.ravel = Raveler(p.weights, p.scalings)
        weights, scalings = self.ravel.to_dicts(vector)
        self.parameters['weights'] = weights
        self.parameters['scalings'] = scalings

    def get_loss(self, vector):
        """
        Method to be called by the regression master.
        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the loss (cost) function.
        """
        return self.lossfunction.f(vector)

    def get_lossprime(self, vector):
        """
        Method to be called by the regression master.
        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the derivative of the loss function
        with respect to model parameters.
        """
        return self.lossfunction.fprime(vector)

    @property
    def lossfunction(self):
        """Allows the user to set a custom loss function. For example,
        >>> from amp.model import LossFunction
        >>> lossfxn = LossFunction(energy_tol=0.0001)
        >>> calc.model.lossfunction = lossfxn
        """
        return self._lossfunction

    @lossfunction.setter
    def lossfunction(self, lossfunction):
        if hasattr(lossfunction, 'attach_model'):
            lossfunction.attach_model(self)  # Allows access to methods.
        self._lossfunction = lossfunction

    def get_atomic_energy(self, afp, index=None, symbol=None,):
        """
        Given input to the neural network, output (which corresponds to energy)
        is calculated about the specified atom. The sum of these for all
        atoms is the total energy (in atom-centered mode).

        :param index: Index of the atom for which atomic energy is calculated
                      (only used in the fingerprinting scheme)
        :type index: int

        :param symbol: Index of the atom for which atomic energy is calculated
                       (only used in the fingerprinting scheme)
        :type symbol: str

        :returns: float -- energy
        """
        p = self.parameters
        self.o[index] = {}
        hiddenlayers = p.hiddenlayers[symbol]
        weight = p.weights[symbol]
        activation = p.activation
        fprange = self.parameters.fprange[symbol]
        # Scale the fingerprints to be in [-1, 1] range.
        for _ in xrange(np.shape(afp)[0]):
            if (fprange[_, 1] - fprange[_, 0]) > (10.**(-8.)):
                afp[_] = -1.0 + 2.0 * ((afp[_] - fprange[_, 0]) /
                                       (fprange[_, 1] - fprange[_, 0]))
        # Calculate node values.
        o = {}  # node values
        layer = 1  # input layer
        net = {}  # excitation
        ohat = {}  # FIXME/ap need description

        len_of_afp = len(afp)
        temp = np.zeros((1, len_of_afp + 1))
        for _ in xrange(len_of_afp):
            temp[0, _] = afp[_]
        temp[0, len(afp)] = 1.0
        ohat[0] = temp
        net[1] = np.dot(ohat[0], weight[1])
        if activation == 'linear':
            o[1] = net[1]  # linear activation
        elif activation == 'tanh':
            o[1] = np.tanh(net[1])  # tanh activation
        elif activation == 'sigmoid':  # sigmoid activation
            o[1] = 1. / (1. + np.exp(-net[1]))
        temp = np.zeros((1, np.shape(o[1])[1] + 1))
        bound = np.shape(o[1])[1]
        for _ in xrange(bound):
            temp[0, _] = o[1][0, _]
        temp[0, np.shape(o[1])[1]] = 1.0
        ohat[1] = temp
        for hiddenlayer in hiddenlayers[1:]:
            layer += 1
            net[layer] = np.dot(ohat[layer - 1], weight[layer])
            if activation == 'linear':
                o[layer] = net[layer]  # linear activation
            elif activation == 'tanh':
                o[layer] = np.tanh(net[layer])  # tanh activation
            elif activation == 'sigmoid':
                # sigmoid activation
                o[layer] = 1. / (1. + np.exp(-net[layer]))
            temp = np.zeros((1, np.size(o[layer]) + 1))
            bound = np.size(o[layer])
            for _ in xrange(bound):
                temp[0, _] = o[layer][0, _]
            temp[0, np.size(o[layer])] = 1.0
            ohat[layer] = temp
        layer += 1  # output layer
        net[layer] = np.dot(ohat[layer - 1], weight[layer])
        if activation == 'linear':
            o[layer] = net[layer]  # linear activation
        elif activation == 'tanh':
            o[layer] = np.tanh(net[layer])  # tanh activation
        elif activation == 'sigmoid':
            # sigmoid activation
            o[layer] = 1. / (1. + np.exp(-net[layer]))

        del hiddenlayers, weight, ohat, net

        len_of_afp = len(afp)
        temp = np.zeros((1, len_of_afp))  # FIXME/ap Need descriptive name
        for _ in xrange(len_of_afp):
            temp[0, _] = afp[_]

        atomic_amp_energy = p.scalings[symbol]['slope'] * \
            float(o[layer]) + p.scalings[symbol]['intercept']
        self.o[index] = o
        self.o[index][0] = temp
        return atomic_amp_energy

    def get_atomic_force(self, direction, derafp, index=None, symbol=None,):
        """
        Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        :param direction: Direction of force.
        :type direction: int
        :param derafp: List of derivatives of inputs
        :type derafp: list
        :param index: Index of the neighbor atom which force is acting at.
                        (only used in the fingerprinting scheme)
        :type index: int
        :param symbol: Symbol of the neighbor atom which force is acting at.
                         (only used in the fingerprinting scheme)
        :type symbol: str

        :returns: float -- force
        """

        p = self.parameters
        o = self.o[index]
        hiddenlayers = p.hiddenlayers[symbol]
        weight = p.weights[symbol]
        scaling = p.scalings[symbol]
        activation = p.activation
        fprange = self.parameters.fprange[symbol]

        # Scaling derivative of fingerprints.
        for _ in xrange(len(derafp)):
            if (fprange[_, 1] - fprange[_, 0]) > (10.**(-8.)):
                derafp[_] = 2.0 * (derafp[_] / (fprange[_, 1] - fprange[_, 0]))

        der_o = {}  # node values
        der_o[0] = derafp
        layer = 0  # input layer
        for hiddenlayer in hiddenlayers[0:]:
            layer += 1
            temp = np.dot(np.matrix(der_o[layer - 1]),
                          np.delete(weight[layer], -1, 0))
            der_o[layer] = [None] * np.size(o[layer])
            bound = np.size(o[layer])
            for j in xrange(bound):
                if activation == 'linear':  # linear function
                    der_o[layer][j] = float(temp[0, j])
                elif activation == 'sigmoid':  # sigmoid function
                    der_o[layer][j] = float(temp[0, j]) * \
                        float(o[layer][0, j] * (1. - o[layer][0, j]))
                elif activation == 'tanh':  # tanh function
                    der_o[layer][j] = float(temp[0, j]) * \
                        float(1. - o[layer][0, j] * o[layer][0, j])
        layer += 1  # output layer
        temp = np.dot(np.matrix(der_o[layer - 1]),
                      np.delete(weight[layer], -1, 0))
        if activation == 'linear':  # linear function
            der_o[layer] = float(temp)
        elif activation == 'sigmoid':  # sigmoid function
            der_o[layer] = float(o[layer] *
                                 (1. - o[layer]) * temp)
        elif activation == 'tanh':  # tanh function
            der_o[layer] = float((1. - o[layer] *
                                  o[layer]) * temp)

        der_o[layer] = [der_o[layer]]

        force = float(-(scaling['slope'] * der_o[layer][0]))

        return force

    def get_dEnergy_dParameters(self, index=None, symbol=None):
        """
        Returns the derivative of energy square error with respect to
        variables.

        :param index: Index of the atom for which atomic energy is calculated
                      (only used in the fingerprinting scheme)
        :type index: int
        :param symbol: Index of the atom for which atomic energy is calculated
                       (only used in the fingerprinting scheme)
        :type symbol: str

        :returns: list of float -- the value of the derivative of energy square
                                   error with respect to variables.
        """
        p = self.parameters
        weights = p.weights
        scalings = p.scalings
        activation = p.activation

        dEnergy_dParameters = np.zeros(self.ravel.count)

        dEnergy_dWeights, dEnergy_dScalings = \
            self.ravel.to_dicts(dEnergy_dParameters)

        self.W = {}
        for elm in weights.keys():
            self.W[elm] = {}
            weight = weights[elm]
            for _ in xrange(len(weight)):
                self.W[elm][_ + 1] = np.delete(weight[_ + 1], -1, 0)

        o = self.o[index]
        W = self.W[symbol]

        N = len(o) - 2  # number of hiddenlayers
        D = {}
        for k in xrange(N + 2):
            D[k] = np.zeros(shape=(np.size(o[k]), np.size(o[k])))
            for j in xrange(np.size(o[k])):
                if activation == 'linear':  # linear
                    D[k][j, j] = 1.
                elif activation == 'sigmoid':  # sigmoid
                    D[k][j, j] = float(o[k][0, j]) * \
                        float((1. - o[k][0, j]))
                elif activation == 'tanh':  # tanh
                    D[k][j, j] = float(1. - o[k][0, j] * o[k][0, j])
        # Calculating delta
        delta = {}
        # output layer
        delta[N + 1] = D[N + 1]
        # hidden layers

        for k in xrange(N, 0, -1):  # backpropagate starting from output layer
            delta[k] = np.dot(D[k], np.dot(W[k + 1], delta[k + 1]))
        # Calculating ohat
        ohat = {}
        for k in xrange(1, N + 2):
            bound = np.size(o[k - 1])
            ohat[k - 1] = np.zeros(shape=(1, bound + 1))
            for j in xrange(bound):
                ohat[k - 1][0, j] = o[k - 1][0, j]
            ohat[k - 1][0, bound] = 1.0

        dEnergy_dScalings[symbol]['intercept'] = 1.
        dEnergy_dScalings[symbol]['slope'] = float(o[N + 1])

        for k in xrange(1, N + 2):
            dEnergy_dWeights[symbol][k] = float(scalings[symbol]['slope']) * \
                np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)

        dEnergy_dParameters = \
            self.ravel.to_vector(dEnergy_dWeights, dEnergy_dScalings)

        self.D[index] = D
        self.delta[index] = delta
        self.ohat[index] = ohat

        return dEnergy_dParameters


# Auxiliary functions #########################################################


def get_random_weights(hiddenlayers, activation, no_of_atoms=None,
                       fprange=None):
    """
    Generates random weight arrays from variables.

    :param hiddenlayers: Dictionary of chemical element symbols and
                         architectures of their corresponding hidden layers of
                         the conventional neural network. Number of nodes of
                         last layer is always one corresponding to energy.
                         However, number of nodes of first layer is equal to
                         three times number of atoms in the system in the case
                         of no descriptor, and is equal to length of symmetry
                         functions in the fingerprinting scheme. Can be fed as:

                         >>> hiddenlayers = (3, 2,)

                         for example, in which a neural network with two hidden
                         layers, the first one having three nodes and the
                         second one having two nodes is assigned (to the whole
                         atomic system in the case of no descriptor, and to
                         each chemical element in the fingerprinting scheme).
                         In the fingerprinting scheme, neural network for each
                         species can be assigned seperately, as:

                         >>> hiddenlayers = {"O":(3,5), "Au":(5,6)}

                         for example.
    :type hiddenlayers: dict

    :param activation: Assigns the type of activation funtion. "linear" refers
                       to linear function, "tanh" refers to tanh function, and
                       "sigmoid" refers to sigmoid function.
    :type activation: str

    :param no_of_atoms: Number of atoms in atomic systems; used only in the
                        case of no descriptor.
    :type no_of_atoms: int

    :param Gs: Dictionary of symbols and lists of dictionaries for making
               symmetry functions. Either auto-genetrated, or given in the
               following form, for example:

               >>> Gs = {"O": [{"type":"G2", "element":"O", "eta":10.},
               ...             {"type":"G4", "elements":["O", "Au"],
               ...              "eta":5., "gamma":1., "zeta":1.0}],
               ...       "Au": [{"type":"G2", "element":"O", "eta":2.},
               ...              {"type":"G4", "elements":["O", "Au"],
               ...               "eta":2., "gamma":1., "zeta":5.0}]}

               Used in the fingerprinting scheme only.
    :type Gs: dict

    :param elements: List of atom symbols; used in the fingerprinting scheme
                     only.
    :type elements: list of str

    :returns: weights
    """
    if activation == 'linear':
        arg_range = 0.3
    else:
        arg_range = 3.

    weight = {}
    nn_structure = {}

    if no_of_atoms is not None:  # pure atomic-coordinates scheme

        if isinstance(hiddenlayers, int):
            nn_structure = ([3 * no_of_atoms] + [hiddenlayers] + [1])
        else:
            nn_structure = (
                [3 * no_of_atoms] +
                [layer for layer in hiddenlayers] + [1])
        weight = {}
        normalized_arg_range = arg_range / (3 * no_of_atoms)
        weight[1] = np.random.random((3 * no_of_atoms + 1,
                                      nn_structure[1])) * \
            normalized_arg_range - \
            normalized_arg_range / 2.
        len_of_hiddenlayers = len(list(nn_structure)) - 3
        for layer in xrange(len_of_hiddenlayers):
            normalized_arg_range = arg_range / \
                nn_structure[layer + 1]
            weight[layer + 2] = np.random.random(
                (nn_structure[layer + 1] + 1,
                 nn_structure[layer + 2])) * \
                normalized_arg_range - normalized_arg_range / 2.
        normalized_arg_range = arg_range / nn_structure[-2]
        weight[len(list(nn_structure)) - 1] = \
            np.random.random((nn_structure[-2] + 1, 1)) \
            * normalized_arg_range - normalized_arg_range / 2.
        len_of_weight = len(weight)
        for _ in xrange(len_of_weight):  # biases
            size = weight[_ + 1][-1].size
            for __ in xrange(size):
                weight[_ + 1][-1][__] = 0.

    else:
        elements = fprange.keys()

        for element in sorted(elements):
            len_of_fps = len(fprange[element])
            if isinstance(hiddenlayers[element], int):
                nn_structure[element] = ([len_of_fps] +
                                         [hiddenlayers[element]] + [1])
            else:
                nn_structure[element] = (
                    [len_of_fps] +
                    [layer for layer in hiddenlayers[element]] + [1])
            weight[element] = {}
            normalized_arg_range = arg_range / len(fprange[element])
            weight[element][1] = np.random.random((len(fprange[element]) + 1,
                                                   nn_structure[
                                                   element][1])) * \
                normalized_arg_range - \
                normalized_arg_range / 2.
            len_of_hiddenlayers = len(list(nn_structure[element])) - 3
            for layer in xrange(len_of_hiddenlayers):
                normalized_arg_range = arg_range / \
                    nn_structure[element][layer + 1]
                weight[element][layer + 2] = np.random.random(
                    (nn_structure[element][layer + 1] + 1,
                     nn_structure[element][layer + 2])) * \
                    normalized_arg_range - normalized_arg_range / 2.
            normalized_arg_range = arg_range / nn_structure[element][-2]
            weight[element][len(list(nn_structure[element])) - 1] = \
                np.random.random((nn_structure[element][-2] + 1, 1)) \
                * normalized_arg_range - normalized_arg_range / 2.

            len_of_weight = len(weight[element])
            for _ in xrange(len_of_weight):  # biases
                size = weight[element][_ + 1][-1].size
                for __ in xrange(size):
                    weight[element][_ + 1][-1][__] = 0.

    return weight


def get_random_scalings(images, activation, elements=None):
    """
    Generates initial scaling matrices, such that the range of activation
    is scaled to the range of actual energies.

    :param images: ASE atoms objects (the training set).
    :type images: dict

    :param activation: Assigns the type of activation funtion. "linear" refers
                       to linear function, "tanh" refers to tanh function, and
                       "sigmoid" refers to sigmoid function.
    :type activation: str

    :param elements: List of atom symbols; used in the fingerprinting scheme
                     only.
    :type elements: list of str

    :returns: scalings
    """
    hashs = images.keys()
    no_of_images = len(hashs)

    max_act_energy = max(image.get_potential_energy(apply_constraint=False)
                         for hash, image in images.items())
    min_act_energy = min(image.get_potential_energy(apply_constraint=False)
                         for hash, image in images.items())

    for count in xrange(no_of_images):
        hash = hashs[count]
        image = images[hash]
        no_of_atoms = len(image)
        if image.get_potential_energy(apply_constraint=False) == \
                max_act_energy:
            no_atoms_of_max_act_energy = no_of_atoms
        if image.get_potential_energy(apply_constraint=False) == \
                min_act_energy:
            no_atoms_of_min_act_energy = no_of_atoms

    max_act_energy_per_atom = max_act_energy / no_atoms_of_max_act_energy
    min_act_energy_per_atom = min_act_energy / no_atoms_of_min_act_energy

    scaling = {}

    if elements is None:  # pure atomic-coordinates scheme

        scaling = {}
        if activation == 'sigmoid':  # sigmoid activation function
            scaling['intercept'] = min_act_energy_per_atom
            scaling['slope'] = (max_act_energy_per_atom -
                                min_act_energy_per_atom)
        elif activation == 'tanh':  # tanh activation function
            scaling['intercept'] = (max_act_energy_per_atom +
                                    min_act_energy_per_atom) / 2.
            scaling['slope'] = (max_act_energy_per_atom -
                                min_act_energy_per_atom) / 2.
        elif activation == 'linear':  # linear activation function
            scaling['intercept'] = (max_act_energy_per_atom +
                                    min_act_energy_per_atom) / 2.
            scaling['slope'] = (10. ** (-10.)) * \
                (max_act_energy_per_atom -
                 min_act_energy_per_atom) / 2.

    else:  # fingerprinting scheme

        for element in elements:
            scaling[element] = {}
            if activation == 'sigmoid':  # sigmoid activation function
                scaling[element]['intercept'] = min_act_energy_per_atom
                scaling[element]['slope'] = (max_act_energy_per_atom -
                                             min_act_energy_per_atom)
            elif activation == 'tanh':  # tanh activation function
                scaling[element]['intercept'] = (max_act_energy_per_atom +
                                                 min_act_energy_per_atom) / 2.
                scaling[element]['slope'] = (max_act_energy_per_atom -
                                             min_act_energy_per_atom) / 2.
            elif activation == 'linear':  # linear activation function
                scaling[element]['intercept'] = (max_act_energy_per_atom +
                                                 min_act_energy_per_atom) / 2.
                scaling[element]['slope'] = (10. ** (-10.)) * \
                                            (max_act_energy_per_atom -
                                             min_act_energy_per_atom) / 2.

    return scaling


class Raveler:

    """Class to ravel and unravel variable values into a single vector.
    This is used for feeding into the optimizer. Feed in a list of
    dictionaries to initialize the shape of the transformation. Note no
    data is saved in the class; each time it is used it is passed either
    the dictionaries or vector. The dictionaries for initialization should
    be two levels deep.
        weights, scalings: variables to ravel and unravel
    """

    def __init__(self, weights, scalings):

        self.count = 0
        self.weightskeys = []
        self.scalingskeys = []
        for key1 in sorted(weights.keys()):  # element
            for key2 in sorted(weights[key1].keys()):  # layer
                value = weights[key1][key2]
                self.weightskeys.append({'key1': key1,
                                         'key2': key2,
                                         'shape': np.array(value).shape,
                                         'size': np.array(value).size})
                self.count += np.array(weights[key1][key2]).size
        for key1 in sorted(scalings.keys()):  # element
            for key2 in sorted(scalings[key1].keys()):  # slope / intercept
                self.scalingskeys.append({'key1': key1,
                                          'key2': key2})
                self.count += 1
        self.vector = np.zeros(self.count)

    def to_vector(self, weights, scalings):
        """Puts the weights and scalings embedded dictionaries into a single
        vector and returns it. The dictionaries need to have the identical
        structure to those it was initialized with."""

        vector = np.zeros(self.count)
        count = 0
        for k in sorted(self.weightskeys):
            lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in sorted(self.scalingskeys):
            vector[count] = scalings[k['key1']][k['key2']]
            count += 1
        return vector

    def to_dicts(self, vector):
        """Puts the vector back into weights and scalings dictionaries of the
        form initialized. vector must have same length as the output of
        unravel."""

        assert len(vector) == self.count
        count = 0
        weights = OrderedDict()
        scalings = OrderedDict()
        for k in sorted(self.weightskeys):
            if k['key1'] not in weights.keys():
                weights[k['key1']] = OrderedDict()
            matrix = vector[count:count + k['size']]
            matrix = matrix.flatten()
            matrix = np.matrix(matrix.reshape(k['shape']))
            weights[k['key1']][k['key2']] = matrix
            count += k['size']
        for k in sorted(self.scalingskeys):
            if k['key1'] not in scalings.keys():
                scalings[k['key1']] = OrderedDict()
            scalings[k['key1']][k['key2']] = vector[count]
            count += 1
        return weights, scalings
