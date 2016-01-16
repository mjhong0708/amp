import numpy as np
from collections import OrderedDict

from ase.calculators.calculator import Parameters

from ..regression import Regressor
from . import LossFunction, calculate_fingerprints_range

try:
    from amp import fmodules
except ImportError:
    fmodules = None



class NeuralNetwork:

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
                 version=None):

        # Version check, particularly if restarting.
        compatibleversions = ['2015.12',]
        if (version is not None) and version not in compatibleversions:
            raise RuntimeError('Error: Trying to use NeuralNetwork'
                               ' version %s, but this module only supports'
                               ' versions %s. You may need an older or '
                               'newer version of Amp.' %
                               (version, compatibleversions))
        else:
            version = compatibleversions[-1]

        # The parameters dictionary contains the minimum information
        # to produce a compatible descriptor; that is, one that gives
        # the same fingerprint when fed an ASE image.
        p = self.parameters = Parameters(
                {'importname': '.model.neuralnetwork.NeuralNetwork',})
        p.version = version
        self.parameters['hiddenlayers'] = hiddenlayers
        self.parameters['weights'] = weights
        self.parameters['scalings'] = scalings
        self.parameters['fprange'] = fprange
        self.parameters['activation'] = activation
        self.parameters['mode'] = mode

        # Checking that the activation function is given correctly:
        if activation not in ['linear', 'tanh', 'sigmoid']:
            raise NotImplementedError('Unknown activation function; '
                                      'activation must be one of '
                                      '"linear", "tanh", or "sigmoid".')

        # Items to be used for training.
        self.regressor = None
        if regressor is not None:
            self.set_regressor(regressor)

    def tostring(self):
        """Returns an evaluatable representation of the calculator that can
        be used to restart the calculator."""
        return self.parameters.tostring()

    def fit(self, trainingimages, fingerprints, log, cores, fortran):
        """Fit the model parameters such that the fingerprints can be used to describe
        the energies in trainingimages. log is the logging object.
        fingerprints is a descriptor object, as would be in calc.fp.
        """
        self.cores = cores
        self.initialize_for_regression(log=log, images=trainingimages, fp=fingerprints)
        if self.regressor is None:
            self.set_regressor()
        
        self.reset_energy()  #FIXME/ap I shouldn't need this.
        result = self.regressor.regress(model=self, log=log)
        return result  # True / False

    def set_regressor(self, regressor=None):
        """Allows the user to set the regression method. Can use the regressor
        can be an instance of amp.regression.Regressor with desired keywords,
        or anything else behaving similarly. FIXME/ap: Define what this means.
        Defaults to using amp.regression.Regressor
        """
        if regressor is None:
            regressor = Regressor()
        self.regressor = regressor

    def get_vector(self):
        """Returns the model parameters (weights, scaling for each network)
        as a single vector."""
        if self.parameters['weights'] is None:
            return None
        return self.ravel.to_vector(weights=self.parameters['weights'],
                                    scalings=self.parameters['scalings'])

    def set_vector(self, vector):
        """Allows the setting of the model parameters (weights, scalings for
        each network) with a single vector."""
        if not hasattr(self, 'ravel'):
            #FIXME/ap I put this in as a kludge for parallel since it hadn't
            # been set in initilalize_for_training in the worker processes
            # since that would change the fprange. Make sure it makes sense here
            # It seems like it should be safe, but maybe some length checking
            # needs to be added to RavelParameters
            p = self.parameters
            self.ravel = RavelVariables(p.weights, p.scalings)
        weights, scalings = self.ravel.to_dicts(vector)
        self.parameters['weights'] = weights
        self.parameters['scalings'] = scalings

    def get_loss(self, vector):
        """
        Method to be called by the regression master.
        Takes one and only one input, a vector of paramters.
        Returns one output, the value of the loss (cost) function.
        """
        if self._lossfunction is None:
            # First run; initialize.
            self.set_loss_function(LossFunction(cores=self.cores))
        return self._lossfunction(vector)

    def set_loss_function(self, function):
        """Allows the user to set a custom loss function. For example,
        >>> from amp.model import CostFunction
        >>> costfxn = CostFunction(energytol=0.0001)
        >>> calc.model.set_loss_function(costfxn)
        """
        function.attach_model(self)  # Allows access to methods.
        self._lossfunction = function

    def ravel_variables(self):
        """
        Wrapper function for raveling weights and scalings into a list.

        :returns: param: Object containing regression's properties.
        """
        if (self.param.regression._variables is None) and self._weights:
            self.param.regression._variables = \
                self.ravel.to_vector(self._weights, self._scalings)

        return self.param

    def reset_energy(self):
        """
        Resets local variables corresponding to energy.
        """
        # FIXME/ap: I don't think we need this function. If we do, it needs to
        # be made clear to the user what it is for. E.g., do you get a different
        # answer from model.get_energy() before and after calling model.reset_energy()?
        self.o = {}
        self.D = {}
        self.delta = {}
        self.ohat = {}

    def reset_forces(self):
        """
        Resets local variables corresponding to forces.
        """
        self.der_coordinates_o = {}

    def update_variables(self, param):
        """
        Updates variables.

        :param param: Object containing regression's properties.
        :type param: ASE calculator's Parameters class
        """
        self._variables = param.regression._variables
        self._weights, self._scalings = \
            self.ravel.to_dicts(self._variables)

        self.W = {}

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            weight = self._weights
            len_of_weight = len(weight)
            j = 0
            while j < len_of_weight:
                self.W[j + 1] = np.delete(weight[j + 1], -1, 0)
                j += 1
        else:  # fingerprinting scheme
            for element in self.elements:
                self.W[element] = {}
                weight = self._weights[element]
                len_of_weight = len(weight)
                j = 0
                while j < len_of_weight:
                    self.W[element][j + 1] = np.delete(weight[j + 1], -1, 0)
                    j += 1

    def introduce_variables(self, log, param):
        """
        Introducing new variables.

        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object
        :param param: Object containing regression's properties.
        :type param: ASE calculator's Parameters class
        """
        log('Introducing new hidden-layer nodes...')

        self._weights, self._scalings = \
            self.ravel.to_dicts(param.regression._variables)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            len_of_weights = len(self._weights)
            j = 1
            while j < len_of_weights + 1:
                shape = np.shape(self._weights[j])
                if j == 1:
                    self._weights[j] = \
                        np.insert(self._weights[j],
                                  shape[1],
                                  shape[0] * [0],
                                  1)
                elif j == len(self._weights):
                    self._weights[j] = \
                        np.insert(self._weights[j],
                                  -1,
                                  shape[1] * [0],
                                  0)
                else:
                    self._weights[j] = \
                        np.insert(self._weights[j],
                                  shape[1],
                                  shape[0] * [0],
                                  1)
                    self._weights[j] = \
                        np.insert(self._weights[j],
                                  -1,
                                  (shape[1] + 1) * [0],
                                  0)
                j += 1

        else:  # fingerprinting scheme
            for element in self.elements:
                len_of_weights = len(self._weights[element])
                j = 1
                while j < len_of_weights + 1:
                    shape = np.shape(self._weights[element][j])
                    if j == 1:
                        self._weights[element][j] = \
                            np.insert(self._weights[element][j],
                                      shape[1],
                                      shape[0] * [0],
                                      1)
                    elif j == len(self._weights[element]):
                        self._weights[element][j] = \
                            np.insert(self._weights[element][j],
                                      -1,
                                      shape[1] * [0],
                                      0)
                    else:
                        self._weights[element][j] = \
                            np.insert(self._weights[element][j],
                                      shape[1],
                                      shape[0] * [0],
                                      1)
                        self._weights[element][j] = \
                            np.insert(self._weights[element][j],
                                      -1,
                                      (shape[1] + 1) * [0],
                                      0)
                    j += 1

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            len_of_hiddenlayers = len(self.hiddenlayers)
            _ = 0
            while _ < len_of_hiddenlayers:
                self.hiddenlayers[_] += 1
                _ += 1
            self.ravel = _RavelVariables(hiddenlayers=self.hiddenlayers,
                                         no_of_atoms=self.no_of_atoms)
        else:  # fingerprinting scheme
            for element in self.elements:
                len_of_hiddenlayers = len(self.hiddenlayers[element])
                _ = 0
                while _ < len_of_hiddenlayers:
                    self.hiddenlayers[element][_] += 1
                    _ += 1
            self.ravel = _RavelVariables(hiddenlayers=self.hiddenlayers,
                                         elements=self.elements,
                                         Gs=param.descriptor.Gs)

        self._variables = \
            self.ravel.to_vector(self._weights, self._scalings)

        param.regression._variables = self._variables

        log('Hidden-layer structure:')
        if param.descriptor is None:  # pure atomic-coordinates scheme
            log(' %s' % str(self.hiddenlayers))
        else:  # fingerprinting scheme
            for item in self.hiddenlayers.items():
                log(' %2s: %s' % item)

        param.regression.hiddenlayers = self.hiddenlayers
        self.hiddenlayers = self.hiddenlayers

        return param

    def get_energy(self, fingerprint):
        """Returns the model-predicted energy for an image, based on its
        fingerprint.
        """
        self.reset_energy()  #FIXME/ap Not sure why/if we need this.
        energy = 0.0
        p = self.parameters
        if p['mode'] == 'image-centered':
            #FIXME/ap: This hasn't been updated for image-centered mode.
            raise NotImplementedError()
        elif p['mode'] == 'atom-centered':
            for index, (element, atomicfingerprint) in enumerate(fingerprint):
                atom_energy = self.get_atomic_energy(input=atomicfingerprint,
                                                     index=index,
                                                     symbol=element)
                energy += atom_energy
        return energy

    def get_atomic_energy(self, input, index=None, symbol=None,):
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
        #FIXME/ap: There is a lot to think through in this part.
        # It doesn't seem like we should need to save self.o,
        # which are the node values, unless this has something
        # to do with making force calls cheaper. We also shouldn't
        # need to pass it the atomic index.
        #FIXME/ap: Should change input, a protected python word,
        # to something else like atomicfingerprint. Or ridge.
        p = self.parameters
        if p.mode == 'image-centered':
            #FIXME/ap Needs updating.
            self.o = {}
            hiddenlayers = self.hiddenlayers
            weight = self._weights
        elif p.mode == 'atom-centered':
            self.o[index] = {}
            hiddenlayers = p.hiddenlayers[symbol]
            weight = p.weights[symbol]
            fprange = self.parameters.fprange[symbol]

        # Scale the fingerprints to be in [-1, 1] range.
        input = -1.0 + 2.0 * ((np.array(input) - fprange[:,0]) / 
                               (fprange[:,1] - fprange[:,0]))

        # Calculate node values.
        o = {}  # node values
        layer = 1  # input layer
        net = {}  # excitation FIXME/ap What?
        ohat = {}  # FIXME/ap What?

        len_of_input = len(input)
        temp = np.zeros((1, len_of_input + 1))
        _ = 0
        while _ < len_of_input:
            temp[0, _] = input[_]
            _ += 1
        temp[0, len(input)] = 1.0
        ohat[0] = temp
        net[1] = np.dot(ohat[0], weight[1])
        if p.activation == 'linear':
            o[1] = net[1]  # linear activation
        elif p.activation == 'tanh':
            o[1] = np.tanh(net[1])  # tanh activation
        elif p.activation == 'sigmoid':  # sigmoid activation
            o[1] = 1. / (1. + np.exp(-net[1]))
        temp = np.zeros((1, np.shape(o[1])[1] + 1))
        bound = np.shape(o[1])[1]
        _ = 0
        while _ < bound:
            temp[0, _] = o[1][0, _]
            _ += 1
        temp[0, np.shape(o[1])[1]] = 1.0
        ohat[1] = temp
        for hiddenlayer in hiddenlayers[1:]:
            layer += 1
            net[layer] = np.dot(ohat[layer - 1], weight[layer])
            if p.activation == 'linear':
                o[layer] = net[layer]  # linear activation
            elif p.activation == 'tanh':
                o[layer] = np.tanh(net[layer])  # tanh activation
            elif p.activation == 'sigmoid':
                # sigmoid activation
                o[layer] = 1. / (1. + np.exp(-net[layer]))
            temp = np.zeros((1, np.size(o[layer]) + 1))
            bound = np.size(o[layer])
            _ = 0
            while _ < bound:
                temp[0, _] = o[layer][0, _]
                _ += 1
            temp[0, np.size(o[layer])] = 1.0
            ohat[layer] = temp
        layer += 1  # output layer
        net[layer] = np.dot(ohat[layer - 1], weight[layer])
        if p.activation == 'linear':
            o[layer] = net[layer]  # linear activation
        elif p.activation == 'tanh':
            o[layer] = np.tanh(net[layer])  # tanh activation
        elif p.activation == 'sigmoid':
            # sigmoid activation
            o[layer] = 1. / (1. + np.exp(-net[layer]))

        del hiddenlayers, weight, ohat, net

        len_of_input = len(input)
        temp = np.zeros((1, len_of_input))
        _ = 0
        while _ < len_of_input:
            temp[0, _] = input[_]
            _ += 1

        if p.mode == 'image-centered':
            #FIXME/ap Needs updating.

            amp_energy = self._scalings['slope'] * \
                float(o[layer]) + self._scalings['intercept']
            self.o = o
            self.o[0] = temp
            return amp_energy

        elif p.mode == 'atom-centered':
            atomic_amp_energy = p.scalings[symbol]['slope'] * \
                float(o[layer]) + p.scalings[symbol]['intercept']
            self.o[index] = o
            self.o[index][0] = temp
            return atomic_amp_energy

    def get_force(self, i, der_indexfp, n_index=None, n_symbol=None,):
        """
        Given derivative of input to the neural network, derivative of output
        (which corresponds to forces) is calculated.

        :param i: Direction of force.
        :type i: int
        :param der_indexfp: List of derivatives of inputs
        :type der_indexfp: list
        :param n_index: Index of the neighbor atom which force is acting at.
                        (only used in the fingerprinting scheme)
        :type n_index: int
        :param n_symbol: Symbol of the neighbor atom which force is acting at.
                         (only used in the fingerprinting scheme)
        :type n_symbol: str

        :returns: float -- force
        """
        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            o = self.o
            hiddenlayers = self.hiddenlayers
            weight = self._weights
        else:  # fingerprinting scheme
            o = self.o[n_index]
            hiddenlayers = self.hiddenlayers[n_symbol]
            weight = self._weights[n_symbol]

        der_o = {}  # node values
        der_o[0] = der_indexfp
        layer = 0  # input layer
        for hiddenlayer in hiddenlayers[0:]:
            layer += 1
            temp = np.dot(np.matrix(der_o[layer - 1]),
                          np.delete(weight[layer], -1, 0))
            der_o[layer] = [None] * np.size(o[layer])
            bound = np.size(o[layer])
            j = 0
            while j < bound:
                if self.activation == 'linear':  # linear function
                    der_o[layer][j] = float(temp[0, j])
                elif self.activation == 'sigmoid':  # sigmoid function
                    der_o[layer][j] = float(temp[0, j]) * \
                        float(o[layer][0, j] * (1. - o[layer][0, j]))
                elif self.activation == 'tanh':  # tanh function
                    der_o[layer][j] = float(temp[0, j]) * \
                        float(1. - o[layer][0, j] * o[layer][0, j])
                j += 1
        layer += 1  # output layer
        temp = np.dot(np.matrix(der_o[layer - 1]),
                      np.delete(weight[layer], -1, 0))
        if self.activation == 'linear':  # linear function
            der_o[layer] = float(temp)
        elif self.activation == 'sigmoid':  # sigmoid function
            der_o[layer] = float(o[layer] *
                                 (1. - o[layer]) * temp)
        elif self.activation == 'tanh':  # tanh function
            der_o[layer] = float((1. - o[layer] *
                                  o[layer]) * temp)

        der_o[layer] = [der_o[layer]]

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            self.der_coordinates_o[i] = der_o
            force = float(-(self._scalings['slope'] * der_o[layer][0]))
        else:  # fingerprinting scheme
            self.der_coordinates_o[(n_index, i)] = der_o
            force = float(-(self._scalings[n_symbol]['slope'] *
                            der_o[layer][0]))

        return force

    def get_variable_der_of_energy(self, index=None, symbol=None):
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
        partial_der_variables_square_error = np.zeros(self.ravel.count)

        partial_der_weights_square_error, partial_der_scalings_square_error = \
            self.ravel.to_dicts(partial_der_variables_square_error)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            o = self.o
            W = self.W
        else:  # fingerprinting scheme
            o = self.o[index]
            W = self.W[symbol]

        N = len(o) - 2  # number of hiddenlayers
        D = {}
        k = 1
        while k < N + 2:
            D[k] = np.zeros(shape=(np.size(o[k]), np.size(o[k])))
            for j in range(np.size(o[k])):
                if self.activation == 'linear':  # linear
                    D[k][j, j] = 1.
                elif self.activation == 'sigmoid':  # sigmoid
                    D[k][j, j] = float(o[k][0, j]) * \
                        float((1. - o[k][0, j]))
                elif self.activation == 'tanh':  # tanh
                    D[k][j, j] = float(1. - o[k][0, j] * o[k][0, j])
            k += 1
        # Calculating delta
        delta = {}
        # output layer
        delta[N + 1] = D[N + 1]
        # hidden layers
        k = N
        while k > 0:  # backpropagate starting from output layer
            delta[k] = np.dot(D[k], np.dot(W[k + 1], delta[k + 1]))
            k -= 1
        # Calculating ohat
        ohat = {}
        k = 1
        while k < N + 2:
            ohat[k - 1] = np.zeros(shape=(1, np.size(o[k - 1]) + 1))
            bound = np.size(o[k - 1])
            j = 0
            while j < bound:
                ohat[k - 1][0, j] = o[k - 1][0, j]
                j += 1
            ohat[k - 1][0, np.size(o[k - 1])] = 1.0
            k += 1

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            partial_der_scalings_square_error['intercept'] = 1.
            partial_der_scalings_square_error['slope'] = float(o[N + 1])
            k = 1
            while k < N + 2:
                partial_der_weights_square_error[k] = \
                    float(self._scalings['slope']) * \
                    np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)
                k += 1
        else:  # fingerprinting scheme
            partial_der_scalings_square_error[symbol]['intercept'] = 1.
            partial_der_scalings_square_error[symbol]['slope'] = \
                float(o[N + 1])
            k = 1
            while k < N + 2:
                partial_der_weights_square_error[symbol][k] = \
                    float(self._scalings[symbol]['slope']) * \
                    np.dot(np.matrix(ohat[k - 1]).T, np.matrix(delta[k]).T)
                k += 1
        partial_der_variables_square_error = \
            self.ravel.to_vector(partial_der_weights_square_error,
                                 partial_der_scalings_square_error)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            self.D = D
            self.delta = delta
            self.ohat = ohat
        else:  # fingerprinting scheme
            self.D[index] = D
            self.delta[index] = delta
            self.ohat[index] = ohat

        return partial_der_variables_square_error

    def get_variable_der_of_forces(self, self_index, i,
                                   n_index=None, n_symbol=None,):
        """
        Returns the derivative of force square error with respect to variables.

        :param self_index: Index of the center atom.
        :type self_index: int
        :param i: Direction of force.
        :type i: int
        :param n_index: Index of the neighbor atom which force is acting at.
                        (only used in the fingerprinting scheme)
        :type n_index: int
        :param n_symbol: Symbol of the neighbor atom which force is acting at.
                         (only used in the fingerprinting scheme)
        :type n_symbol: str

        :returns: list of float -- the value of the derivative of force square
                                   error with respect to variables.
        """
        partial_der_variables_square_error = np.zeros(self.ravel.count)

        partial_der_weights_square_error, partial_der_scalings_square_error = \
            self.ravel.to_dicts(partial_der_variables_square_error)

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            o = self.o
            der_coordinates_o = self.der_coordinates_o[i]
            W = self.W
            delta = self.delta
            ohat = self.ohat
            D = self.D
        else:  # fingerprinting scheme
            o = self.o[n_index]
            der_coordinates_o = self.der_coordinates_o[(n_index, i)]
            W = self.W[n_symbol]
            delta = self.delta[n_index]
            ohat = self.ohat[n_index]
            D = self.D[n_index]

        N = len(o) - 2
        der_coordinates_D = {}
        k = 1
        while k < N + 2:
            # Calculating coordinate derivative of D matrix
            der_coordinates_D[k] = \
                np.zeros(shape=(np.size(o[k]), np.size(o[k])))
            for j in range(np.size(o[k])):
                if self.activation == 'linear':  # linear
                    der_coordinates_D[k][j, j] = 0.
                elif self.activation == 'tanh':  # tanh
                    der_coordinates_D[k][j, j] = \
                        - 2. * o[k][0, j] * der_coordinates_o[k][j]
                elif self.activation == 'sigmoid':  # sigmoid
                    der_coordinates_D[k][j, j] = der_coordinates_o[k][j] - \
                        2. * o[k][0, j] * der_coordinates_o[k][j]
            k += 1
        # Calculating coordinate derivative of delta
        der_coordinates_delta = {}
        # output layer
        der_coordinates_delta[N + 1] = der_coordinates_D[N + 1]
        # hidden layers
        temp1 = {}
        temp2 = {}
        k = N
        while k > 0:
            temp1[k] = np.dot(W[k + 1], delta[k + 1])
            temp2[k] = np.dot(W[k + 1], der_coordinates_delta[k + 1])
            der_coordinates_delta[k] = \
                np.dot(der_coordinates_D[k], temp1[k]) + np.dot(D[k], temp2[k])
            k -= 1
        # Calculating coordinate derivative of ohat and
        # coordinates weights derivative of atomic_output
        der_coordinates_ohat = {}
        der_coordinates_weights_atomic_output = {}
        k = 1
        while k < N + 2:
            der_coordinates_ohat[k - 1] = \
                [None] * (1 + len(der_coordinates_o[k - 1]))
            count = 0
            bound = len(der_coordinates_o[k - 1])
            while count < bound:
                der_coordinates_ohat[k - 1][count] = \
                    der_coordinates_o[k - 1][count]
                count += 1
            der_coordinates_ohat[k - 1][count] = 0.
            der_coordinates_weights_atomic_output[k] = \
                np.dot(np.matrix(der_coordinates_ohat[k - 1]).T,
                       np.matrix(delta[k]).T) + \
                np.dot(np.matrix(ohat[k - 1]).T,
                       np.matrix(der_coordinates_delta[k]).T)
            k += 1

        if self.param.descriptor is None:  # pure atomic-coordinates scheme
            k = 1
            while k < N + 2:
                partial_der_weights_square_error[k] = \
                    float(self._scalings['slope']) * \
                    der_coordinates_weights_atomic_output[k]
                k += 1
            partial_der_scalings_square_error['slope'] = \
                der_coordinates_o[N + 1][0]
        else:  # fingerprinting scheme
            k = 1
            while k < N + 2:
                partial_der_weights_square_error[n_symbol][k] = \
                    float(self._scalings[n_symbol]['slope']) * \
                    der_coordinates_weights_atomic_output[k]
                k += 1
            partial_der_scalings_square_error[n_symbol]['slope'] = \
                der_coordinates_o[N + 1][0]
        partial_der_variables_square_error = \
            self.ravel.to_vector(partial_der_weights_square_error,
                                 partial_der_scalings_square_error)

        return partial_der_variables_square_error

    def initialize_for_regression(self, log, images, fp):
        """
        Prints out in the log file and generates variables if do not already 
        exist. This is essentially storing data for the regression.

        :param log: Write function at which to log data. Note this must be a
                    callable function.
        :type log: Logger object
        :param param: Object containing regression's properties.
        :type param: ASE calculator's Parameters class
        :param elements: List of atom symbols.
        :type elements: list of str
        :param images: ASE atoms objects (the training set).
        :type images: dict

        :returns: Object containing regression's properties.
        """
        #FIXME/ap: Fold into 'fit'?

        # FIXME/ap: This could be good to keep for the parallel version,
        # but we definitely don't want it adjusting fprange then.
        if not hasattr(self, '_lossfunction'):
            self._lossfunction = None  # Will hold a method.

        p = self.parameters
        tp = self.trainingparameters = Parameters()
        tp['trainingimages'] = images
        tp['fingerprint'] = fp
        tp['log'] = log

        p['mode'] = fp.parameters['mode']
        log(' Regression in %s mode.' % p.mode)

        if p['mode'] == 'atom-centered':
            self.elements = fp.parameters.elements
        elif p['mode'] == 'image-centered':
            self.elements = None
        
        if 'fprange' not in p or p.fprange is None:
            log('Calculating new fingerprint range.')
            p['fprange'] = calculate_fingerprints_range(fp, images)

        if p['mode'] == 'image-centered':  # pure atomic-coordinates scheme

            #FIXME/ap This part needs updating.
            raise NotImplementedError()
            self.no_of_atoms = param.no_of_atoms
            self.hiddenlayers = param.regression.hiddenlayers

            structure = self.hiddenlayers
            if isinstance(structure, str):
                structure = structure.split('-')
            elif isinstance(structure, int):
                structure = [structure]
            else:
                structure = list(structure)
            hiddenlayers = [int(part) for part in structure]
            self.hiddenlayers = hiddenlayers

            self.ravel = _RavelVariables(
                hiddenlayers=self.hiddenlayers,
                elements=self.elements,
                no_of_atoms=self.no_of_atoms)

        elif p['mode'] == 'atom-centered':  # fingerprinting scheme

            # If hiddenlayers is fed by the user in the tuple format,
            # it will now be converted to a dictionary.
            if isinstance(p.hiddenlayers, tuple):
                hiddenlayers = {}
                for element in self.elements:
                    hiddenlayers[element] = p.hiddenlayers
                p.hiddenlayers = hiddenlayers


            # FIXME/ap I deleted some lines that converted the interior
            # tuple to a list. Can't figure out why this would be needed.
            # Maybe when growing the NN?

            if False:
                self.ravel = _RavelVariables(hiddenlayers=p.hiddenlayers,
                                             elements=self.elements,
                                             Gs=fp.parameters.Gs)
            # FIXME/ap This was moved later, and back to old version.


            # FIXME/ap: delete Gs... those aren't fit variables here.
            # This is especially important as not all descriptors use
            # this nomenclature.

            # FIXME/ap: Especially important as Gs are not a required
            # attribute of descriptors. This will make it so that this
            # method is incompatible with anything but Behler.

        log('Hidden-layer structure:')
        if p.mode == 'image-centered':
            log(' %s' % str(p.hiddenlayers))
        elif p.mode == 'atom-centered':
            for item in p.hiddenlayers.items():
                log(' %2s: %s' % item)

        if p.weights is None:
            log('Initializing with random weights.')
            if p.mode == 'image-centered':
                #FIXME/ap need to update this block.
                raise NotImplementedError()
                self._weights = make_weight_matrices(self.hiddenlayers,
                                                     self.activation,
                                                     self.no_of_atoms)
            elif p.mode == 'atom-centered':
                p.weights = make_weight_matrices(p.hiddenlayers,
                                                 p.activation,
                                                 None,
                                                 fp.parameters.Gs,
                                                 self.elements)

            # FIXME/ap: Again, delete Gs in above.
 
        else:
            log('Initial weights already present.')
        # If scalings are not given, generates random scalings
        if not p.scalings:
            log('Initializing with random scalings.')

            if p.mode == 'image-centered':
                #FIXME/ap need to update this block.
                raise NotImplementedError()
                self._scalings = make_scalings_matrices(images,
                                                        self.activation,)
            elif p.mode == 'atom-centered':
                p.scalings = make_scalings_matrices(images,
                                                    p.activation,
                                                    self.elements,)
        else:
            log('Initial scalings already present.')


        self.ravel = RavelVariables(weights=p.weights, scalings=p.scalings)


    def send_data_to_fortran(self, param):
        """
        Sends regression data to fortran.

        :param param: Object containing symmetry function's (if any) and
                      regression's properties.
        :type param: ASE calculator's Parameters class
        """
        if param.descriptor is None:
            fingerprinting = False
        else:
            fingerprinting = True

        if fingerprinting:
            no_layers_of_elements = \
                [3 if isinstance(param.regression.hiddenlayers[elm], int)
                 else (len(param.regression.hiddenlayers[elm]) + 2)
                 for elm in self.elements]
            nn_structure = OrderedDict()
            for elm in self.elements:
                if isinstance(param.regression.hiddenlayers[elm], int):
                    nn_structure[elm] = ([len(param.descriptor.Gs[elm])] +
                                         [param.regression.hiddenlayers[elm]] +
                                         [1])
                else:
                    nn_structure[elm] = ([len(param.descriptor.Gs[elm])] +
                                         [layer for layer in
                                          param.regression.hiddenlayers[elm]] +
                                         [1])

            no_nodes_of_elements = [nn_structure[elm][_]
                                    for elm in self.elements
                                    for _ in range(len(nn_structure[elm]))]

        else:
            no_layers_of_elements = []
            if isinstance(param.regression.hiddenlayers, int):
                no_layers_of_elements = [3]
            else:
                no_layers_of_elements = \
                    [len(param.regression.hiddenlayers) + 2]
            if isinstance(param.regression.hiddenlayers, int):
                nn_structure = ([3 * param.no_of_atoms] +
                                [param.regression.hiddenlayers] + [1])
            else:
                nn_structure = ([3 * param.no_of_atoms] +
                                [layer for layer in
                                 param.regression.hiddenlayers] + [1])
            no_nodes_of_elements = [nn_structure[_]
                                    for _ in range(len(nn_structure))]

        fmodules.regression.no_layers_of_elements = no_layers_of_elements
        fmodules.regression.no_nodes_of_elements = no_nodes_of_elements
        if param.regression.activation == 'tanh':
            activation_signal = 1
        elif param.regression.activation == 'sigmoid':
            activation_signal = 2
        elif param.regression.activation == 'linear':
            activation_signal = 3
        fmodules.regression.activation_signal = activation_signal

###############################################################################
###############################################################################
###############################################################################


def make_weight_matrices(hiddenlayers, activation, no_of_atoms=None, Gs=None,
                         elements=None):
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
        layer = 0
        while layer < len_of_hiddenlayers:
            normalized_arg_range = arg_range / \
                nn_structure[layer + 1]
            weight[layer + 2] = np.random.random(
                (nn_structure[layer + 1] + 1,
                 nn_structure[layer + 2])) * \
                normalized_arg_range - normalized_arg_range / 2.
            layer += 1
        normalized_arg_range = arg_range / nn_structure[-2]
        weight[len(list(nn_structure)) - 1] = \
            np.random.random((nn_structure[-2] + 1, 1)) \
            * normalized_arg_range - normalized_arg_range / 2.
        len_of_weight = len(weight)
        _ = 0
        while _ < len_of_weight:  # biases
            size = weight[_ + 1][-1].size
            __ = 0
            while __ < size:
                weight[_ + 1][-1][__] = 0.
                __ += 1
            _ += 1

    else:

        for element in sorted(elements):
            len_of_fps = len(Gs[element])
            if isinstance(hiddenlayers[element], int):
                nn_structure[element] = ([len_of_fps] +
                                         [hiddenlayers[element]] + [1])
            else:
                nn_structure[element] = (
                    [len_of_fps] +
                    [layer for layer in hiddenlayers[element]] + [1])
            weight[element] = {}
            normalized_arg_range = arg_range / len(Gs[element])
            weight[element][1] = np.random.random((len(Gs[element]) + 1,
                                                   nn_structure[
                                                   element][1])) * \
                normalized_arg_range - \
                normalized_arg_range / 2.
            len_of_hiddenlayers = len(list(nn_structure[element])) - 3
            layer = 0
            while layer < len_of_hiddenlayers:
                normalized_arg_range = arg_range / \
                    nn_structure[element][layer + 1]
                weight[element][layer + 2] = np.random.random(
                    (nn_structure[element][layer + 1] + 1,
                     nn_structure[element][layer + 2])) * \
                    normalized_arg_range - normalized_arg_range / 2.
                layer += 1
            normalized_arg_range = arg_range / nn_structure[element][-2]
            weight[element][len(list(nn_structure[element])) - 1] = \
                np.random.random((nn_structure[element][-2] + 1, 1)) \
                * normalized_arg_range - normalized_arg_range / 2.

            len_of_weight = len(weight[element])
            _ = 0
            while _ < len_of_weight:  # biases
                size = weight[element][_ + 1][-1].size
                __ = 0
                while __ < size:
                    weight[element][_ + 1][-1][__] = 0.
                    __ += 1
                _ += 1

    return weight


def make_scalings_matrices(images, activation, elements=None):
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

    count = 0
    while count < no_of_images:
        hash = hashs[count]
        image = images[hash]
        no_of_atoms = len(image)
        if image.get_potential_energy(apply_constraint=False) == \
                max_act_energy:
            no_atoms_of_max_act_energy = no_of_atoms
        if image.get_potential_energy(apply_constraint=False) == \
                min_act_energy:
            no_atoms_of_min_act_energy = no_of_atoms
        count += 1

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

    del images

    return scaling

###############################################################################
###############################################################################
###############################################################################

class RavelVariables:

    """Class to ravel and unravel variable values into a single vector.
    This is used for feeding into the optimizer. Feed in a list of
    dictionaries to initialize the shape of the transformation. Note no
    data is saved in the class; each time it is used it is passed either
    the dictionaries or vector. The dictionaries for initialization should
    be two levels deep.
        weights, scalings: variables to ravel and unravel
    """

    ##########################################################################

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
        self.der_of_weights_norm = np.zeros(self.count)

    #########################################################################

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

    #########################################################################

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



class _RavelVariables:

    """
    Class to ravel and unravel weight and scaling values into a single vector.
    This is used for feeding into the optimizer. Feed in a list of
    dictionaries to initialize the shape of the transformation. Note that no
    data is saved in the class; each time it is used it is passed either
    the dictionaries or vector.

    :param hiddenlayers: Dictionary of chemical element symbols and
                        architectures of their corresponding hidden layers of
                        the conventional neural network.
    :type hiddenlayers: dict
    :param elements: List of atom symbols; used in the fingerprinting scheme
                     only.
    :type elements: list of str
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
    :param no_of_atoms: Number of atoms in atomic systems; used only in the
                        case of no descriptor.
    :type no_of_atoms: int
    """
    ###########################################################################

    def __init__(self, hiddenlayers, elements=None, Gs=None, no_of_atoms=None):
        raise NotImplementedError('Fixme: put this back to the old version?')

        self._no_of_atoms = no_of_atoms
        self._vectorlength= 0
        self._weightskeys = []
        self._scalingskeys = []

        if self._no_of_atoms is None:  # fingerprinting scheme

            for element in elements:
                len_of_hiddenlayers = len(hiddenlayers[element])
                layer = 1
                while layer < len_of_hiddenlayers + 2:
                    if layer == 1:
                        shape = \
                            (len(Gs[element]) + 1, hiddenlayers[element][0])
                    elif layer == (len(hiddenlayers[element]) + 1):
                        shape = (hiddenlayers[element][layer - 2] + 1, 1)
                    else:
                        shape = (
                            hiddenlayers[element][layer - 2] + 1,
                            hiddenlayers[element][layer - 1])
                    size = shape[0] * shape[1]
                    self._weightskeys.append({'key1': element,
                                              'key2': layer,
                                              'shape': shape,
                                              'size': size})
                    self._vectorlength += size
                    layer += 1

            for element in elements:
                self._scalingskeys.append({'key1': element,
                                           'key2': 'intercept'})
                self._scalingskeys.append({'key1': element,
                                           'key2': 'slope'})
                self._vectorlength += 2

        else:  # pure atomic-coordinates scheme

            len_of_hiddenlayers = len(hiddenlayers)
            layer = 1
            while layer < len_of_hiddenlayers + 2:
                if layer == 1:
                    shape = (3 * no_of_atoms + 1, hiddenlayers[0])
                elif layer == (len(hiddenlayers) + 1):
                    shape = (hiddenlayers[layer - 2] + 1, 1)
                else:
                    shape = (
                        hiddenlayers[layer - 2] + 1, hiddenlayers[layer - 1])
                size = shape[0] * shape[1]
                self._weightskeys.append({'key': layer,
                                          'shape': shape,
                                          'size': size})
                self._vectorlength += size
                layer += 1

            self._scalingskeys.append({'key': 'intercept'})
            self._scalingskeys.append({'key': 'slope'})
            self._vectorlength += 2

    ###########################################################################

    def to_vector(self, weights, scalings):
        """
        Puts the weights and scalings embedded dictionaries into a single
        vector and returns it. The dictionaries need to have the identical
        structure to those it was initialized with.

        :param weights: In the case of no descriptor, keys correspond to
                        layers and values are two dimensional arrays of network
                        weight. In the fingerprinting scheme, keys correspond
                        to chemical elements and values are dictionaries with
                        layer keys and network weight two dimensional arrays as
                        values. Arrays are set up to connect node i in the
                        previous layer with node j in the current layer with
                        indices w[i,j]. The last value for index i corresponds
                        to bias. If weights is not given, arrays will be
                        randomly generated.
        :type weights: dict
        :param scalings: In the case of no descriptor, keys are "intercept"
                         and "slope" and values are real numbers. In the
                         fingerprinting scheme, keys correspond to chemical
                         elements and values are dictionaries with "intercept"
                         and "slope" keys and real number values. If scalings
                         is not given, it will be randomly generated.
        :type scalings: dict

        :returns: List of variables
        """
        vector = np.zeros(self._vectorlength)
        count = 0
        for k in sorted(self._weightskeys):
            if self._no_of_atoms is None:  # fingerprinting scheme
                lweights = np.array(weights[k['key1']][k['key2']]).ravel()
            else:  # pure atomic-coordinates scheme
                lweights = (np.array(weights[k['key']])).ravel()
            vector[count:(count + lweights.size)] = lweights
            count += lweights.size
        for k in sorted(self._scalingskeys):
            if self._no_of_atoms is None:  # fingerprinting scheme
                vector[count] = scalings[k['key1']][k['key2']]
            else:  # pure atomic-coordinates scheme
                vector[count] = scalings[k['key']]
            count += 1
        return vector

    ###########################################################################

    def to_dicts(self, vector):
        """
        Puts the vector back into weights and scalings dictionaries of the
        form initialized. vector must have same length as the output of
        unravel.

        :param vector: List of variables.
        :type vector: list

        :returns: weights and scalings
        """
        assert len(vector) == self._vectorlength
        count = 0
        weights = OrderedDict()
        scalings = OrderedDict()
        for k in sorted(self._weightskeys):
            if self._no_of_atoms is None:  # fingerprinting scheme
                if k['key1'] not in weights.keys():
                    weights[k['key1']] = OrderedDict()
            matrix = vector[count:count + k['size']]
            matrix = np.array(matrix).flatten()
            matrix = np.matrix(matrix.reshape(k['shape']))
            if self._no_of_atoms is None:  # fingerprinting scheme
                weights[k['key1']][k['key2']] = matrix
            else:  # pure atomic-coordinates scheme
                weights[k['key']] = matrix
            count += k['size']
        for k in sorted(self._scalingskeys):
            if self._no_of_atoms is None:  # fingerprinting scheme
                if k['key1'] not in scalings.keys():
                    scalings[k['key1']] = OrderedDict()
                scalings[k['key1']][k['key2']] = vector[count]
            else:  # pure atomic-coordinates scheme
                scalings[k['key']] = vector[count]
            count += 1
        return weights, scalings

