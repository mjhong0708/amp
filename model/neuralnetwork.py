import numpy as np
from collections import OrderedDict
import os
from ase.calculators.calculator import Parameters
from ..regression import Regressor
from ..model import LossFunction, calculate_fingerprints_range
from ..model import Model
from ..utilities import Logger, hash_images, make_filename


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
                         chemical element in the atom-centered mode).
                         In the atom-centered mode, neural network for each
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
                    In the atom-centered mode, keys correspond to chemical
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

    :param fprange: Range of fingerprints of each chemical species.
                    Should be fed as a dictionary of chemical
                    species and a list of minimum and maximun, e.g:

                    >>> fprange={"Pd": [0.31, 0.59], "O":[0.56, 0.72]}

    :type fprange: dict

    :param regressor: Regressor object for finding best fit model parameters,
                      e.g. by loss function optimization via
                      amp.regression.Regressor.
    :type regressor: object

    :param mode: Can be either 'atom-centered' or 'image-centered'.
    :type mode: str

    :param lossfunction: Loss function object, if at all desired by the user.
    :type lossfunction: object

    :param version: Version of this class.
    :type version: object

    :param fortran: If True, allows for extrapolation, if False, does not
                    allow.
    :type fortran: bool

    .. note:: Dimensions of weight two dimensional arrays should be consistent
              with hiddenlayers.

    :raises: RuntimeError, NotImplementedError
    """

    def __init__(self, hiddenlayers=(5, 5), activation='tanh', weights=None,
                 scalings=None, fprange=None, regressor=None, mode=None,
                 lossfunction=None, version=None, fortran=True):

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
        self.fortran = fortran

    def fit(self,
            trainingimages,
            descriptor,
            log,
            cores,
            only_setup=False,
            ):
        """Fit the model parameters such that the fingerprints can be used to
        describe the energies in trainingimages. log is the logging object.
        descriptor is a descriptor object, as would be in calc.descriptor.

        :param trainingimages: Hashed dictionary of training images.
        :type trainingimages: dict

        :param descriptor: Class representing local atomic environment.
        :type descriptor: object

        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object

        :param cores: Number of cores to parallelize over. If not specified,
                      attempts to determine from environment.
        :type cores: int

        :param only_setup: only_setup is primarily for debugging.
                           It initializes all variables but skips the last
                           line of starting the regressor.
        :type only_setup: bool
        """

        # Set all parameters and report to logfile.
        self.cores = cores

        if self.lossfunction is None:
            self.lossfunction = LossFunction(cores=self.cores)
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
        log('Regression in %s mode.' % p.mode)

        if 'fprange' not in p or p.fprange is None:
            log('Calculating new fingerprint range; this range is part '
                'of the model.')
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
            for item in p.hiddenlayers.iteritems():
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

        if only_setup:
            return

        # Regress the model.
        self.step = 0
        result = self.regressor.regress(model=self, log=log)
        return result  # True / False

    @property
    def vector(self):
        """Access to get or set the model parameters (weights, scaling for
        each network) as a single vector, useful in particular for
        regression.
        """
        if self.parameters['weights'] is None:
            return None
        p = self.parameters
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights, p.scalings)
        return self.ravel.to_vector(weights=p.weights, scalings=p.scalings)

    @vector.setter
    def vector(self, vector):
        """
        :param vector: Parameters of the regression model in the form of a
                       list.
        :type vector: list
        """
        p = self.parameters
        if not hasattr(self, 'ravel'):
            self.ravel = Raveler(p.weights, p.scalings)
        weights, scalings = self.ravel.to_dicts(vector)
        p['weights'] = weights
        p['scalings'] = scalings

    def get_loss(self, vector):
        """
        Method to be called by the regression master.
        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the loss (cost) function.

        :param vector: Parameters of the regression model in the form of a
                       list.
        :type vector: list
        """
        if self.step == 0:
            filename = make_filename(self.parent.label,
                                     '-initial-parameters.amp')
            filename = self.parent.save(filename, overwrite=True)
        elif self.step % 100 == 0:
            path = os.path.join(self.parent.label + '-checkpoints/')
            if self.step == 100:
                if not os.path.exists(path):
                    os.mkdir(path)
            self.parent.log('Saving checkpoint data.')
            filename = make_filename(path,
                                     'parameters-checkpoint-%d.amp' % self.step)
            filename = self.parent.save(filename, overwrite=True)
        self.step += 1
        return self.lossfunction.get_loss(vector, lossprime=False)['loss']

    def get_lossprime(self, vector):
        """
        Method to be called by the regression master.
        Takes one and only one input, a vector of parameters.
        Returns one output, the value of the derivative of the loss function
        with respect to model parameters.

        :param vector: Parameters of the regression model in the form of a
                       list.
        :type vector: list
        """
        return self.lossfunction.get_loss(vector,
                                          lossprime=True)['dloss_dparameters']

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
        """
        :param lossfunction: Loss function object, if at all desired by the
                             user.
        :type lossfunction: object
        """
        if hasattr(lossfunction, 'attach_model'):
            lossfunction.attach_model(self)  # Allows access to methods.
        self._lossfunction = lossfunction

    def get_atomic_energy(self, afp, index, symbol,):
        """
        Given input to the neural network, output (which corresponds to energy)
        is calculated about the specified atom. The sum of these for all
        atoms is the total energy (in atom-centered mode).

        :param afp: Atomic fingerprints in the form of a list to be used as
                    input to the neural network.
        :type afp: list

        :param index: Index of the atom for which atomic energy is calculated
                      (only used in the atom-centered mode)
        :type index: int

        :param symbol: Symbol of the atom for which atomic energy is calculated
                       (only used in the atom-centered mode)
        :type symbol: str

        :returns: float -- energy
        """
        assert self.parameters.mode == 'atom-centered', \
            'get_atomic_energy should only be called in atom-centered mode.'

        scaling = self.parameters.scalings[symbol]
        outputs = calculate_nodal_outputs(self.parameters, afp, symbol,)
        atomic_amp_energy = scaling['slope'] * \
            float(outputs[len(outputs) - 1]) + \
            scaling['intercept']

        return atomic_amp_energy

    def get_force(self, afp, derafp,
                  direction,
                  nindex=None, nsymbol=None,):
        """
        Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        :param afp: Atomic fingerprints in the form of a list to be used as
                    input to the neural network.
        :type afp: list

        :param derafp: Derivatives of atomic fingerprints in the form of a list
                       to be used as input to the neural network.
        :type derafp: list

        :param direction: Direction of force.
        :type direction: int

        :param nindex: Index of the neighbor atom which force is acting at.
                        (only used in the atom-centered mode)
        :type nindex: int

        :param nsymbol: Symbol of the neighbor atom which force is acting at.
                         (only used in the atom-centered mode)
        :type nsymbol: str

        :returns: float -- force
        """

        scaling = self.parameters.scalings[nsymbol]
        outputs = calculate_nodal_outputs(self.parameters, afp, nsymbol,)
        dOutputs_dInputs = calculate_dOutputs_dInputs(self.parameters, derafp,
                                                      outputs, nsymbol,)

        force = float((scaling['slope'] *
                       dOutputs_dInputs[len(dOutputs_dInputs) - 1][0]))
        # force is multiplied by -1, because it is -dE/dx and not dE/dx.
        force *= -1.

        return force

    def get_dAtomicEnergy_dParameters(self, afp, index=None, symbol=None):
        """
        Returns the derivative of energy square error with respect to
        variables.

        :param afp: Atomic fingerprints in the form of a list to be used as
                    input to the neural network.
        :type afp: list

        :param index: Index of the atom for which atomic energy is calculated
                      (only used in the atom-centered mode)
        :type index: int
        :param symbol: Symbol of the atom for which atomic energy is calculated
                       (only used in the atom-centered mode)
        :type symbol: str

        :returns: list of float -- the value of the derivative of energy square
                                   error with respect to variables.
        """
        p = self.parameters
        scaling = p.scalings[symbol]
        # self.W dictionary initiated.
        self.W = {}
        for elm in p.weights.keys():
            self.W[elm] = {}
            weight = p.weights[elm]
            for _ in xrange(len(weight)):
                self.W[elm][_ + 1] = np.delete(weight[_ + 1], -1, 0)
        W = self.W[symbol]

        dAtomicEnergy_dParameters = np.zeros(self.ravel.count)
        dAtomicEnergy_dWeights, dAtomicEnergy_dScalings = \
            self.ravel.to_dicts(dAtomicEnergy_dParameters)

        outputs = calculate_nodal_outputs(self.parameters, afp, symbol,)
        ohat, D, delta = calculate_ohat_D_delta(self.parameters, outputs, W)

        dAtomicEnergy_dScalings[symbol]['intercept'] = 1.
        dAtomicEnergy_dScalings[symbol][
            'slope'] = float(outputs[len(outputs) - 1])
        for k in xrange(1, len(outputs)):
            dAtomicEnergy_dWeights[symbol][k] = float(scaling['slope']) * \
                np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)

        dAtomicEnergy_dParameters = \
            self.ravel.to_vector(
                dAtomicEnergy_dWeights, dAtomicEnergy_dScalings)

        return dAtomicEnergy_dParameters

    def get_dForce_dParameters(self, afp, derafp,
                               direction,
                               nindex=None, nsymbol=None,):
        """
        Returns the derivative of force square error with respect to variables.

        :param afp: Atomic fingerprints in the form of a list to be used as
                    input to the neural network.
        :type afp: list

        :param derafp: Derivatives of atomic fingerprints in the form of a list
                       to be used as input to the neural network.
        :type derafp: list

        :param direction: Direction of force.
        :type direction: int

        :param nindex: Index of the neighbor atom which force is acting at.
                        (only used in the atom-centered mode)
        :type nindex: int

        :param nsymbol: Symbol of the neighbor atom which force is acting at.
                         (only used in the atom-centered mode)
        :type nsymbol: str

        :returns: list of float -- the value of the derivative of force square
                                   error with respect to variables.
        """
        p = self.parameters
        scaling = p.scalings[nsymbol]
        activation = p.activation
        # self.W dictionary initiated.
        self.W = {}
        for elm in p.weights.keys():
            self.W[elm] = {}
            weight = p.weights[elm]
            for _ in xrange(len(weight)):
                self.W[elm][_ + 1] = np.delete(weight[_ + 1], -1, 0)
        W = self.W[nsymbol]

        dForce_dParameters = np.zeros(self.ravel.count)

        dForce_dWeights, dForce_dScalings = \
            self.ravel.to_dicts(dForce_dParameters)

        outputs = calculate_nodal_outputs(self.parameters, afp, nsymbol,)
        ohat, D, delta = calculate_ohat_D_delta(self.parameters, outputs, W)
        dOutputs_dInputs = calculate_dOutputs_dInputs(self.parameters, derafp,
                                                      outputs, nsymbol,)

        N = len(outputs) - 2
        dD_dInputs = {}
        for k in xrange(1, N + 2):
            # Calculating coordinate derivative of D matrix
            dD_dInputs[k] = np.zeros(shape=(np.size(outputs[k]),
                                            np.size(outputs[k])))
            for j in xrange(np.size(outputs[k])):
                if activation == 'linear':  # linear
                    dD_dInputs[k][j, j] = 0.
                elif activation == 'tanh':  # tanh
                    dD_dInputs[k][j, j] = \
                        - 2. * outputs[k][0, j] * dOutputs_dInputs[k][j]
                elif activation == 'sigmoid':  # sigmoid
                    dD_dInputs[k][j, j] = dOutputs_dInputs[k][j] - \
                        2. * outputs[k][0, j] * dOutputs_dInputs[k][j]
        # Calculating coordinate derivative of delta
        dDelta_dInputs = {}
        # output layer
        dDelta_dInputs[N + 1] = dD_dInputs[N + 1]
        # hidden layers
        temp1 = {}
        temp2 = {}
        for k in xrange(N, 0, -1):
            temp1[k] = np.dot(W[k + 1], delta[k + 1])
            temp2[k] = np.dot(W[k + 1], dDelta_dInputs[k + 1])
            dDelta_dInputs[k] = \
                np.dot(dD_dInputs[k], temp1[k]) + np.dot(D[k], temp2[k])
        # Calculating coordinate derivative of ohat and
        # coordinates weights derivative of atomic_output
        dOhat_dInputs = {}
        dOutput_dInputsdWeights = {}
        for k in xrange(1, N + 2):
            dOhat_dInputs[k - 1] = [None] * (1 + len(dOutputs_dInputs[k - 1]))
            bound = len(dOutputs_dInputs[k - 1])
            for count in xrange(bound):
                dOhat_dInputs[k - 1][count] = dOutputs_dInputs[k - 1][count]
            dOhat_dInputs[k - 1][count + 1] = 0.
            dOutput_dInputsdWeights[k] = \
                np.dot(np.matrix(dOhat_dInputs[k - 1]).T,
                       np.matrix(delta[k]).T) + \
                np.dot(np.matrix(ohat[k - 1]).T,
                       np.matrix(dDelta_dInputs[k]).T)

        for k in xrange(1, N + 2):
            dForce_dWeights[nsymbol][k] = float(scaling['slope']) * \
                dOutput_dInputsdWeights[k]
        dForce_dScalings[nsymbol]['slope'] = dOutputs_dInputs[N + 1][0]
        dForce_dParameters = self.ravel.to_vector(dForce_dWeights,
                                                  dForce_dScalings)
        # force is multiplied by -1, because it is -dE/dx and not dE/dx.
        dForce_dParameters *= -1.

        return dForce_dParameters

# Auxiliary functions #########################################################


def calculate_nodal_outputs(parameters, afp, symbol,):
    """
    Given input to the neural network, output (which corresponds to energy)
    is calculated about the specified atom. The sum of these for all
    atoms is the total energy (in atom-centered mode).

    :param parameters: ASE dictionary object.
    :type parameters: dict

    :param afp: Atomic fingerprints in the form of a list to be used as
                input to the neural network.
    :type afp: list

    :param symbol: Symbol of the atom for which atomic energy is calculated
                    (only used in the atom-centered mode)
    :type symbol: str

    :returns: dict -- outputs of neural network nodes
    """

    _afp = np.array(afp).copy()
    hiddenlayers = parameters.hiddenlayers[symbol]
    weight = parameters.weights[symbol]
    activation = parameters.activation

    fprange = parameters.fprange[symbol]
    # Scale the fingerprints to be in [-1, 1] range.
    for _ in xrange(np.shape(_afp)[0]):
        if (fprange[_][1] - fprange[_][0]) > (10.**(-8.)):
            _afp[_] = -1.0 + 2.0 * ((_afp[_] - fprange[_][0]) /
                                    (fprange[_][1] - fprange[_][0]))

    # Calculate node values.
    o = {}  # node values
    layer = 1  # input layer
    net = {}  # excitation
    ohat = {}  # FIXME/ap need description

    len_of_afp = len(_afp)
    temp = np.zeros((1, len_of_afp + 1))
    for _ in xrange(len_of_afp):
        temp[0, _] = _afp[_]
    temp[0, len(_afp)] = 1.0
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

    len_of_afp = len(_afp)
    temp = np.zeros((1, len_of_afp))  # FIXME/ap Need descriptive name
    for _ in xrange(len_of_afp):
        temp[0, _] = _afp[_]
    o[0] = temp

    return o


def calculate_dOutputs_dInputs(parameters, derafp, outputs, nsymbol,):
    """
    :param parameters: ASE dictionary object.
    :type parameters: dict

    :param derafp: Derivatives of atomic fingerprints in the form of a list
                   to be used as input to the neural network.
    :type derafp: list

    :param outputs: Outputs of neural network nodes.
    :type outputs: dict

    :param nsymbol: Symbol of the atom for which atomic energy is calculated
                    (only used in the atom-centered mode)
    :type nsymbol: str

    :returns: dict -- Derivatives of outputs of neural network nodes w.r.t.
                      inputs.
    """

    _derafp = np.array(derafp).copy()
    hiddenlayers = parameters.hiddenlayers[nsymbol]
    weight = parameters.weights[nsymbol]
    activation = parameters.activation

    fprange = parameters.fprange[nsymbol]
    # Scaling derivative of fingerprints.
    for _ in xrange(len(_derafp)):
        if (fprange[_][1] - fprange[_][0]) > (10.**(-8.)):
            _derafp[_] = 2.0 * (_derafp[_] / (fprange[_][1] - fprange[_][0]))

    dOutputs_dInputs = {}  # node values
    dOutputs_dInputs[0] = _derafp
    layer = 0  # input layer
    for hiddenlayer in hiddenlayers[0:]:
        layer += 1
        temp = np.dot(np.matrix(dOutputs_dInputs[layer - 1]),
                      np.delete(weight[layer], -1, 0))
        dOutputs_dInputs[layer] = [None] * np.size(outputs[layer])
        bound = np.size(outputs[layer])
        for j in xrange(bound):
            if activation == 'linear':  # linear function
                dOutputs_dInputs[layer][j] = float(temp[0, j])
            elif activation == 'sigmoid':  # sigmoid function
                dOutputs_dInputs[layer][j] = float(temp[0, j]) * \
                    float(outputs[layer][0, j] * (1. - outputs[layer][0, j]))
            elif activation == 'tanh':  # tanh function
                dOutputs_dInputs[layer][j] = float(temp[0, j]) * \
                    float(1. - outputs[layer][0, j] * outputs[layer][0, j])
    layer += 1  # output layer
    temp = np.dot(np.matrix(dOutputs_dInputs[layer - 1]),
                  np.delete(weight[layer], -1, 0))
    if activation == 'linear':  # linear function
        dOutputs_dInputs[layer] = float(temp)
    elif activation == 'sigmoid':  # sigmoid function
        dOutputs_dInputs[layer] = \
            float(outputs[layer] * (1. - outputs[layer]) * temp)
    elif activation == 'tanh':  # tanh function
        dOutputs_dInputs[layer] = \
            float((1. - outputs[layer] * outputs[layer]) * temp)

    dOutputs_dInputs[layer] = [dOutputs_dInputs[layer]]

    return dOutputs_dInputs


def calculate_ohat_D_delta(parameters, outputs, W):
    """
    Calculates extra matrices ohat, D, delta needed in mathematical
    manipulations. Notations are consistent with those of
    'Rojas, R. Neural Networks - A Systematic Introduction.
    Springer-Verlag, Berlin, first edition 1996'

    :param parameters: ASE dictionary object.
    :type parameters: dict

    :param outputs: Outputs of neural network nodes.
    :type outputs: dict

    :param W: The same as weight dictionary, but the last rows associated with
              biases are deleted in W.
    :type W: dict
    """

    activation = parameters.activation

    N = len(outputs) - 2  # number of hiddenlayers
    D = {}
    for k in xrange(N + 2):
        D[k] = np.zeros(shape=(np.size(outputs[k]), np.size(outputs[k])))
        for j in xrange(np.size(outputs[k])):
            if activation == 'linear':  # linear
                D[k][j, j] = 1.
            elif activation == 'sigmoid':  # sigmoid
                D[k][j, j] = float(outputs[k][0, j]) * \
                    float((1. - outputs[k][0, j]))
            elif activation == 'tanh':  # tanh
                D[k][j, j] = float(1. - outputs[k][0, j] * outputs[k][0, j])
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
        bound = np.size(outputs[k - 1])
        ohat[k - 1] = np.zeros(shape=(1, bound + 1))
        for j in xrange(bound):
            ohat[k - 1][0, j] = outputs[k - 1][0, j]
        ohat[k - 1][0, bound] = 1.0

    return ohat, D, delta


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
                         functions in the atom-centered mode. Can be fed as:

                         >>> hiddenlayers = (3, 2,)

                         for example, in which a neural network with two hidden
                         layers, the first one having three nodes and the
                         second one having two nodes is assigned (to the whole
                         atomic system in the case of no descriptor, and to
                         each chemical element in the atom-centered mode).
                         In the atom-centered mode, neural network for each
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

    :param fprange: Range of fingerprints of each chemical species.
                    Should be fed as a dictionary of chemical
                    species and a list of minimum and maximun, e.g:

                    >>> fprange={"Pd": [0.31, 0.59], "O":[0.56, 0.72]}

    :type fprange: dict

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

    :param elements: List of atom symbols; used in the atom-centered mode
                     only.
    :type elements: list of str

    :returns: scalings
    """
    hashs = images.keys()
    no_of_images = len(hashs)

    max_act_energy = max(image.get_potential_energy(apply_constraint=False)
                         for hash, image in images.iteritems())
    min_act_energy = min(image.get_potential_energy(apply_constraint=False)
                         for hash, image in images.iteritems())

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

    else:  # atom-centered mode

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
            weights[k['key1']][k['key2']] = matrix.tolist()
            count += k['size']
        for k in sorted(self.scalingskeys):
            if k['key1'] not in scalings.keys():
                scalings[k['key1']] = OrderedDict()
            scalings[k['key1']][k['key2']] = vector[count]
            count += 1
        return weights, scalings

# Analysis tools ##############################################################


class NodePlot:

    """Creates plots to visualize the output of the nodes in the neural
    networks.

    initialize with a calculator that has parameters; e.g. a trained
    calculator or else one in which fit has been called with the setup_only
    flag turned on.

    Call with the 'plot' method, which takes as argment a list of images
    """

    def __init__(self, calc):
        self.calc = calc
        self.data = {}  # For accumulating the data.
        # Local imports; these are not package-wide dependencies.
        from matplotlib import pyplot
        from matplotlib.backends.backend_pdf import PdfPages
        self.pyplot = pyplot
        self.PdfPages = PdfPages

    def plot(self, images, filename='nodeplot.pdf'):
        """ Creates a plot of the output of each node, as a violin plot.
        """
        calc = self.calc
        data = {}
        log = Logger('develop.log')
        images = hash_images(images, log=log)
        calc.descriptor.calculate_fingerprints(images=images,
                                               cores=1,
                                               log=log,
                                               calculate_derivatives=False)
        for hash, image in images.iteritems():
            fingerprints = calc.descriptor.fingerprints[hash]
            for fp in fingerprints:
                outputs = calculate_nodal_outputs(calc.model.parameters,
                                                  afp=fp[1],
                                                  symbol=fp[0])
                self._accumulate(symbol=fp[0], output=outputs)

        self._finalize_table()

        with self.PdfPages(filename) as pdf:
            for symbol, data in self.data.iteritems():
                fig = self._makefig(symbol)
                pdf.savefig(fig)
                self.pyplot.close(fig)

    def _makefig(self, symbol, save=False):
        """Makes a figure for one element."""

        fig = self.pyplot.figure(figsize=(8.5, 11.0))
        lm = 0.1
        rm = 0.05
        bm = 0.05
        tm = 0.05
        vg = 0.05
        numplots = 1 + self.data[symbol]['header'][-1][0]
        axwidth = 1. - lm - rm
        axheight = (1. - bm - tm - (numplots - 1) * vg) / numplots

        d = self.data[symbol]
        for layer in range(1 + d['header'][-1][0]):
            ax = fig.add_axes((lm,
                               1. - tm - axheight - (axheight + vg) * layer,
                               axwidth, axheight))
            indices = [_ for _, label in enumerate(d['header'])
                       if label[0] == layer]
            sub = d['table'][:, indices]
            ax.violinplot(dataset=sub, positions=range(len(indices)))
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlim(-0.5, len(indices) - 0.5)
            ax.set_ylabel('Layer %i' % layer)
        ax.set_xlabel('node')
        fig.text(0.5, 1. - 0.5 * tm, 'Node outputs for %s' % symbol,
                 ha='center', va='center')

        if save:
            fig.savefig(save)
        return fig

    def _accumulate(self, symbol, output):
        """Accumulates the data for the symbol."""
        data = self.data
        layerkeys = output.keys()  # Correspond to layers.
        layerkeys.sort()
        if symbol not in data:
            # Create headers, structure.
            data[symbol] = {'header': [],
                            'table': []}
            for layerkey in layerkeys:
                v = output[layerkey]
                v = v.reshape(v.size).tolist()
                data[symbol]['header'].extend([(layerkey, _) for _ in
                                              range(len(v))])
        # Add as a row to data table.
        row = []
        for layerkey in layerkeys:
            v = output[layerkey]
            v = v.reshape(v.size).tolist()
            row.extend(v)
        data[symbol]['table'].append(row)

    def _finalize_table(self):
        """Converts the data table into a numpy array."""
        for symbol in self.data:
            self.data[symbol]['table'] = np.array(self.data[symbol]['table'])
