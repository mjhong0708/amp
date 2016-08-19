# This module was contributed by:
#    Zachary Ulissi
#    Department of Chemical Engineering
#    Stanford University
#    zulissi@gmail.com
# Help/testing/discussions: Andrew Doyle (Stanford) and
# the AMP development team

# This module implements energy- and force- training using Google's
# TensorFlow library. In doing so, the training is multithreaded and GPU
# accelerated.

import numpy as np
import tensorflow as tf
import random
import string
import sklearn.linear_model
import pickle
import uuid
import time
from amp.model import LossFunction
from ..utilities import now,ConvergenceOccurred
#from  tensorflow.contrib.opt.python.training import external_optimizer
from  tensorflow.contrib.opt import ScipyOptimizerInterface
class tfAmpNN:
    """
    TensorFlow-based Neural Network model. (Google's machine-learning
    code).

    Initialize with:
      elementFingerprintLengths: dictionary (one for each element type)
          that contains the fingerprint length for each element

      hiddenlayers: structure of the neural network

      activation: activation type

      keep_prob: dropout rate for the neural network to reduce overfitting.
        (keep_prob=1. uses all nodes, keep_prob~0.5-0.8 better for training)

      RMSEtarget: target for the final loss function
      Note this is the per-image value.

      maxTrainingEpochs: maximum number of times to loop through the
      training data before giving up

      batchsize: batch size for minibatch (if miniBatch is set to True)

      tfVars: tensorflow variables (used if restoring from a previous save)

      saveVariableName: name used for the internal tensorflow variable
      naming scheme.  if variables have the same name as another model in
      the same tensorflow session, there will be collisions

      sess: tensorflow session to use (None means start a new session)

      maxAtomsForces: number of atoms to be used in the force training.  It
      sets the upper bound on the number of atoms that can be used to
      calculate the force for (e.g. if maxAtomsForces=40, then forces can
      only be calculated for images with less than 40 atoms)

      energy_coefficient and force_coefficient are used to adjust the
      loss function; note you must turn on train_forces when calling
      Amp.train (or model.fit) if you want to use force training.

      scikit_model: a pickled version of the scikit model used to
      re-establish this model.
    """

    def __init__(self,
                 elementFingerprintLengths,
                 hiddenlayers=(5, 5),
                 activation='relu',
                 keep_prob=1.,
                 maxTrainingEpochs=10000,
                 batchsize=20,
                 initialTrainingRate=1e-4,
                 miniBatch=True,
                 tfVars=None,
                 saveVariableName=None,
                 parameters=None,
                 sess=None,
                 maxAtomsForces=50,
                 energy_coefficient=1.0,
                 force_coefficient=0.04,
                 scikit_model=None,
                 convergenceCriteria=None,
                 optimizationMethod='ADAM',
                 input_keep_prob=0.8,
                 ADAM_optimizer_params={'beta1':0.9},
                 regularization_strength=None,
                 applyLinearModel=True,
                ):
        
        self.parameters = {} if parameters is None else parameters
        for prop in ['elementFPScales', 'energyMeanScale', 'energyProdScale',
                     'energyPerElement']:
            if prop not in self.parameters:
                self.parameters[prop] = None
        if 'convergence' in self.parameters:
            1
        elif convergenceCriteria is None:
            self.parameters['convergence']={'energy_rmse': 0.001,
                                          'energy_maxresid': None,
                                          'force_rmse': 0.005,
                                          'force_maxresid': None}
        else:
            self.parameters['convergence']=convergenceCriteria
            
        if scikit_model is not None:
            self.linearmodel = pickle.loads(scikit_model)
        else:
            self.linearmodel=None
        if 'applyLinearMode' not in self.parameters:
            self.parameters['applyLinearModel']=applyLinearModel
        if 'energy_coefficient' not in self.parameters:
            self.parameters['energy_coefficient']=energy_coefficient
        if 'force_coefficient' not in self.parameters:
            self.parameters['force_coefficient']=force_coefficient
        if 'ADAM_optimizer_params' not in self.parameters:
            self.parameters['ADAM_optimizer_params']=ADAM_optimizer_params
        if 'regularization_strength' not in self.parameters:
            self.parameters['regularization_strength']=regularization_strength
        self.hiddenlayers = hiddenlayers
        if isinstance(activation, basestring):
            self.activationName = activation
            self.activation = eval('tf.nn.' + activation)
        else:
            self.activation = activation
            self.activationName = activation.__name__
        self.keep_prob = keep_prob
        self.input_keep_prob=input_keep_prob
        self.elements = elementFingerprintLengths.keys()
        self.elements.sort()
        if saveVariableName is None:
            self.saveVariableName = str(uuid.uuid4())[:8]
        else:
            self.saveVariableName = saveVariableName

        self.elementFingerprintLengths={}
        for element in self.elements:
            self.elementFingerprintLengths[element] = elementFingerprintLengths[element]
        
        
        if sess is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = sess

        with self.sess.as_default():
            self.constructModel()
            trainvarlist=tf.trainable_variables()
            trainvarlist=[a for a in trainvarlist if a.name[:8]==self.saveVariableName]
            self.saver = tf.train.Saver(trainvarlist)
            if tfVars is not None:
                self.initializeVariables()
                #trainable_vars = tf.trainable_variables()
                #all_vars = tf.all_variables()
                #untrainable_vars = []
                #for var in all_vars:
                #    if var not in trainable_vars:
                #        untrainable_vars.append(var)
                #    varnamesplit=var.name.split('_')
                #beta_accumulators=self.adam_optimizer_instance._get_beta_accumulators()
                #for var in beta_accumulators:
                #    untrainable_vars.append(var)
                #    if len(varnamesplit)>0 and varnamesplit[1]=='power' and (varnamesplit[0]=='beta1' or varnamesplit[0]=='beta2'):
                #        untrainable_vars.append(var)
                #    #re-initialize the ADAM ops
                with open('tfAmpNN-checkpoint-restore', 'w') as fhandle:
                    fhandle.write(tfVars)
                self.saver.restore(self.sess, 'tfAmpNN-checkpoint-restore')
                #self.sess.run(tf.initialize_variables(untrainable_vars))
            else:
                self.initializeVariables()
        self.maxTrainingEpochs = maxTrainingEpochs
        self.batchsize = batchsize
        self.initialTrainingRate = initialTrainingRate
        self.miniBatch = miniBatch

        #optimizer can be 'ADAM' or 'l-BFGS-b'
        self.optimizationMethod=optimizationMethod
        self.maxAtomsForces = maxAtomsForces

    def constructModel(self):
        """Sets up the tensorflow neural networks for each atom type."""

        # Make tensorflow inputs for each element.
        tensordict = {}
        indsdict = {}
        maskdict = {}
        tensorDerivDict = {}
        for element in self.elements:
            tensordict[element] = tf.placeholder(
                "float", shape=[None, self.elementFingerprintLengths[element]],name='tensor_%s'%element)
            tensorDerivDict[element] = tf.placeholder("float",
                                                      shape=[None, None, 3, self.elementFingerprintLengths[element]],name='tensorderiv_%s'%element)
            indsdict[element] = tf.placeholder("int64", shape=[None],name='indsdict_%s'%element)
            maskdict[element] = tf.placeholder("float", shape=[None, 1],name='maskdict_%s'%element)
        self.indsdict = indsdict
        self.tileDerivs = tf.placeholder("int32", shape=[4],name='tileDerivs')
        self.tensordict = tensordict
        self.maskdict = maskdict
        self.energyProdScale=tf.placeholder("float",name='energyProdScale')
        self.tensorDerivDict = tensorDerivDict

        # y_ is the input energy for each configuration.
        self.y_ = tf.placeholder("float", shape=[None, 1],name='y_')
        self.input_keep_prob_in=tf.placeholder("float",name='input_kee_prob_in')
        self.keep_prob_in = tf.placeholder("float",name='keep_prob_in')
        self.nAtoms_in = tf.placeholder("float", shape=[None, 1],name='nAtoms_in')
        self.batchsizeInput = tf.placeholder("int32",name='batchsizeInput')
        self.learningrate = tf.placeholder("float",name='learningrate')
        self.forces_in = tf.placeholder(
            "float", shape=[None, None, 3], name='forces_in')
        self.energycoefficient = tf.placeholder("float")
        self.forcecoefficient = tf.placeholder("float")

        # Generate a multilayer neural network for each element type.
        outdict = {}
        forcedict = {}
        l2_regularization_dict={}
        for element in self.elements:
            if isinstance(self.hiddenlayers, dict):
                networkListToUse = self.hiddenlayers[element]
            else:
                networkListToUse = self.hiddenlayers
            outdict[element], forcedict[element],l2_regularization_dict[element] = model(tensordict[element],
                                                         indsdict[element],
                                                         self.keep_prob_in,
							 self.input_keep_prob_in,
                                                         self.batchsizeInput,
                                                         networkListToUse,
                                                         self.activation,
                                                         self.elementFingerprintLengths[
                                                             element],
                                                         mask=maskdict[
                                                             element],
                                                         name=self.saveVariableName,
                                                         dxdxik=self.tensorDerivDict[
                                                             element],
                                                         tilederiv=self.tileDerivs,element=element)
        self.outdict = outdict

        # The total energy is the sum of the energies over each atom type.
        keylist = self.elements
        ytot = outdict[keylist[0]]
        for i in range(1, len(keylist)):
            ytot = ytot + outdict[keylist[i]]
        self.energy = ytot*self.energyProdScale

        # The total force is the sum of the forces over each atom type.
        Ftot = forcedict[keylist[0]]
        for i in range(1, len(keylist)):
            Ftot = Ftot + forcedict[keylist[i]]
        self.forces = -Ftot*self.energyProdScale
        
        l2_regularization =l2_regularization_dict[keylist[0]]
        for i in range(1, len(keylist)):
            l2_regularization = l2_regularization + l2_regularization_dict[keylist[i]]
        # Define output nodes for the energy of a configuration, a loss
        # function, and the loss per atom (which is what we usually track)
        #self.loss = tf.sqrt(tf.reduce_sum(
        #    tf.square(tf.sub(self.energy, self.y_))))
        #self.lossPerAtom = tf.reduce_sum(
        #    tf.square(tf.div(tf.sub(self.energy, self.y_), self.nAtoms_in)))
        
        #loss function, as included in model/__init__.py
        self.energy_loss = tf.reduce_sum(
            tf.square(tf.div(tf.sub(self.energy, self.y_), self.nAtoms_in)))
        # Define the training step for energy training.
        

        #self.loss_forces = self.forcecoefficient * \
        #    tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.forces_in, self.forces))))
        #force loss function, as included in model/__init__.py
        self.force_loss= tf.reduce_sum(tf.div(tf.reduce_mean(tf.square(tf.sub(self.forces_in, self.forces)),2),self.nAtoms_in))
        relativeA=0.5
        self.force_loss= tf.reduce_sum(
			    tf.div(
                                tf.reduce_mean(
                                    tf.div(
                                        tf.square(tf.sub(self.forces_in, self.forces)),tf.square(self.forces_in)+relativeA**2.)*relativeA**2.,2),self.nAtoms_in))
        

        #Define max residuals
        self.energy_maxresid=tf.reduce_max(tf.abs(tf.div(tf.sub(self.energy, self.y_), self.nAtoms_in))) 
        self.force_maxresid=tf.reduce_max(tf.abs(tf.sub(self.forces_in, self.forces)))
        
        # Define the training step for force training.
        if self.parameters['regularization_strength'] is not None:
            self.loss = self.forcecoefficient*self.force_loss + self.energycoefficient*self.energy_loss+self.parameters['regularization_strength']*l2_regularization
            self.energy_loss_regularized=self.energy_loss+self.parameters['regularization_strength']*l2_regularization
        else:
            self.loss = self.forcecoefficient*self.force_loss + self.energycoefficient*self.energy_loss
            self.energy_loss_regularized=self.energy_loss
        
        self.adam_optimizer_instance= tf.train.AdamOptimizer(self.learningrate, **self.parameters['ADAM_optimizer_params'])
        self.train_step = self.adam_optimizer_instance.minimize(self.energy_loss_regularized)
        self.train_step_forces =self.adam_optimizer_instance.minimize(self.loss)
        #self.loss_forces_relative = self.forcecoefficient * tf.sqrt(tf.reduce_mean(tf.square(tf.div(tf.sub(self.forces_in, self.forces),self.forces_in+0.0001))))
        #self.force_loss_relative= tf.reduce_sum(tf.div(tf.reduce_mean(tf.div(tf.square(tf.sub(self.forces_in, self.forces)),tf.square(self.forces_in)+0.005**2.),2),self.nAtoms_in))
        #self.loss_relative = self.forcecoefficient*self.loss_forces_relative + self.energycoefficient*self.energy_loss
        #self.train_step_forces = tf.adam_optimizer_instance.minimize(self.loss_relative)

    def initializeVariables(self):
        """Resets all of the variables in the current tensorflow model."""
        self.sess.run(tf.initialize_all_variables())

    def generateFeedInput(self, curinds,
                          energies,
                          atomArraysAll,
                          atomArraysAllDerivs,
                          nAtomsDict,
                          atomsIndsReverse,
                          batchsize,
                          trainingrate,
                          keepprob, inputkeepprob,natoms,
                          forcesExp=0.,
                          forces=False,
                          energycoefficient=1.,
                          forcecoefficient=None):
        """Generates the input dictionary that maps various inputs on
        the python side to placeholders for the tensorflow model.  """

        atomArraysFinal, atomArraysDerivsFinal, atomInds = generateBatch(curinds,
                                                                         self.elements,
                                                                         atomArraysAll,
                                                                         nAtomsDict,
                                                                         atomsIndsReverse,
                                                                         atomArraysAllDerivs)
        feedinput = {}
        tilederivs = (0, 0, 0, 0)
        for element in self.elements:
            if len(atomArraysFinal[element]) > 0:
                feedinput[self.tensordict[element]] = atomArraysFinal[
                    element] / self.parameters['elementFPScales'][element]
                feedinput[self.indsdict[element]] = atomInds[element]
                feedinput[self.maskdict[element]] = np.ones((batchsize, 1))
                if forcecoefficient > 1.e-5:
                    feedinput[self.tensorDerivDict[element]
                              ] = atomArraysDerivsFinal[element]
                if len(atomArraysDerivsFinal[element]) > 0:
                    tilederivs = np.array([1, atomArraysDerivsFinal[element].shape[
                                          1], atomArraysDerivsFinal[element].shape[2], 1])
            else:
                feedinput[self.tensordict[element]] = np.zeros(
                    (1, self.elementFingerprintLengths[element]))
                feedinput[self.indsdict[element]] = [0]
                feedinput[self.maskdict[element]] = np.zeros((batchsize, 1))
                feedinput[self.tensorDerivDict[element]] = np.zeros(
                                                                (1, 1, 3, self.elementFingerprintLengths[element]))
        feedinput[self.tileDerivs] = tilederivs
        feedinput[self.y_] = energies[curinds]
        feedinput[self.batchsizeInput] = batchsize
        feedinput[self.learningrate] = trainingrate
        feedinput[self.keep_prob_in] = keepprob
        feedinput[self.input_keep_prob_in] = inputkeepprob
        feedinput[self.nAtoms_in] = natoms[curinds]
        if forcecoefficient > 1.e-5:
            feedinput[self.forces_in] = forcesExp[curinds]
            feedinput[self.forcecoefficient] = forcecoefficient
            feedinput[self.energycoefficient] = energycoefficient
        feedinput[self.energyProdScale]=self.parameters['energyProdScale']
        return feedinput

    def fit(self, trainingimages, descriptor, cores=1, log=None,
            outlier_energy=10.):
        """Fit takes a bunch of training images (which are assumed to have a
        working calculator attached), and fits the internal variables to the
        training images.
        """

        # The force_coefficient was moved out of Amp.train; pull from the
        # initialization variables. This doesn't catch if the user sends
        # train_forces=False in Amp.train, but an issue is filed to fix
        # this.

        self.log=log
        tempconvergence=self.parameters['convergence'].copy()
        if self.parameters['force_coefficient']<1e-5:
            tempconvergence['force_rmse']=None
            tempconvergence['force_maxresid']=None
        
        
        if self.parameters['force_coefficient']>1e-5:
            lf=LossFunction(convergence=tempconvergence,energy_coefficient=self.parameters['energy_coefficient'],force_coefficient=self.parameters['force_coefficient'],cores=1)
            lf.attach_model(self,images=trainingimages,fingerprints=descriptor.fingerprints,fingerprintprimes=descriptor.fingerprintprimes)
        else:
            lf=LossFunction(convergence=tempconvergence,energy_coefficient=self.parameters['energy_coefficient'],force_coefficient=0.,cores=1)
            lf.attach_model(self,images=trainingimages,fingerprints=descriptor.fingerprints)
        lf._initialize()
        # Inputs:
        # trainingimages:
        batchsize = self.batchsize
        if self.parameters['force_coefficient'] == 0.:
            fingerprintDerDB = None
        else:
            fingerprintDerDB = descriptor.fingerprintprimes
        images = trainingimages
        keylist = images.keys()
        fingerprintDB = descriptor.fingerprints
        self.parameters['numTrainingImages']=len(keylist)
        self.maxAtomsForces = np.max(map(lambda x: len(images[x]), keylist))
        atomArraysAll, nAtomsDict, atomsIndsReverse, natoms, atomArraysAllDerivs = generateTensorFlowArrays(
            fingerprintDB, self.elements, keylist, fingerprintDerDB, self.maxAtomsForces)
        energies = map(lambda x: [images[x].get_potential_energy()], keylist)
        energies = np.array(energies)

        natomsArray = np.zeros((len(keylist), len(self.elements)))
        for i in range(len(images)):
            for j in range(len(self.elements)):
                natomsArray[i][j] = nAtomsDict[self.elements[j]][i]

        # simple model to help normalize the energies by guessing a per-atom
        # energy.  This is helpful to removing the large electronic energy
        # associate for each element, making it easier to regress later
        if self.linearmodel is None and self.parameters['applyLinearModel']:
            model_ransac = sklearn.linear_model.RANSACRegressor(
                 sklearn.linear_model.LinearRegression(), residual_threshold=outlier_energy, min_samples=0.1)
            model_ransac.fit(natomsArray, energies)
            self.linearmodel = model_ransac
        if self.parameters['applyLinearModel']:
            energies = energies - self.linearmodel.predict(natomsArray)

        #newkeylist = []
        #newenergies = []
        #for i in range(len(keylist)):
        #    if self.linearmodel.inlier_mask_[i] == True:
        #        newkeylist.append(keylist[i])
        #        newenergies.append(energies[i])
        
        #keylist = newkeylist
        #energies = np.array(newenergies)
        atomArraysAll, nAtomsDict, atomsIndsReverse, natoms, atomArraysAllDerivs = generateTensorFlowArrays(
            fingerprintDB, self.elements, keylist, fingerprintDerDB, self.maxAtomsForces)

        self.parameters['energyMeanScale'] = np.mean(energies)
        energies = energies - self.parameters['energyMeanScale']
        self.parameters['energyProdScale'] = np.mean(np.abs(energies))
        #energies = energies / self.parameters['energyProdScale']
        self.parameters['elementFPScales'] = {}
        for element in self.elements:
            if len(atomArraysAll[element]) == 0:
                self.parameters['elementFPScales'][element] = 1.
            else:
                self.parameters['elementFPScales'][element] = np.max(
                    np.max(np.abs(atomArraysAll[element])))
        for key in self.parameters['elementFPScales']:
            self.parameters['elementFPScales'][key]=1

        if self.maxAtomsForces >0:
            if self.parameters['force_coefficient'] is not None:
                #forces = map(lambda x: images[x].get_forces(
                #    apply_constraint=False), keylist)
                forces = np.zeros((len(keylist), self.maxAtomsForces, 3))
                for i in range(len(keylist)):
                    atoms = images[keylist[i]]
                    forces[i, 0:len(atoms), :] = atoms.get_forces(
                    apply_constraint=False)
                #forces = forces / self.parameters['energyProdScale']
            else:
                forces = 0.
        else:
            forces=0.

        if not(self.miniBatch):
            batchsize = len(keylist)

        
        def trainmodel(trainingrate, keepprob, inputkeepprob,maxepochs):
            icount = 1
            icount_global = 1
            indlist = np.arange(len(keylist))
            converge_save=[]

            # continue taking training steps as long as we haven't hit the RMSE
            # minimum of the max number of epochs
            while  (icount < maxepochs):

                # if we're in minibatch mode, shuffle the index list
                if self.miniBatch:
                    np.random.shuffle(indlist)

                for i in range(int(len(keylist) / batchsize)):

                    # if we're doing minibatch, construct a new set of inputs
                    if self.miniBatch or (not(self.miniBatch)and(icount == 1)):
                        if self.miniBatch:
                            curinds = indlist[
                                np.arange(batchsize) + i * batchsize]
                        else:
                            curinds = range(len(keylist))

                        feedinput = self.generateFeedInput(curinds,
                                                           energies,
                                                           atomArraysAll,
                                                           atomArraysAllDerivs,
                                                           nAtomsDict,
                                                           atomsIndsReverse,
                                                           batchsize,
                                                           trainingrate,
                                                           keepprob,inputkeepprob,
                                                           natoms,
                                                           forcesExp=forces,
                                                           energycoefficient=self.parameters['energy_coefficient'],
                                                           forcecoefficient=self.parameters['force_coefficient'])

                    # run a training step with the inputs.
                    if (self.parameters['force_coefficient'] < 1.e-5):
                        self.sess.run(self.train_step, feed_dict=feedinput)
                    else:
                        self.sess.run(self.train_step_forces,
                                      feed_dict=feedinput)

                    # Print the loss function every 100 evals.
                    #if (self.miniBatch)and(icount % 100 == 0):
                    #	feed_keepprob_save=feedinput[self.keep_prob_in]
                    #	feed_keepprob_save_input=feedinput[self.input_keep_prob_in]
                    #	feedinput[self.keep_prob_in]=1.
                    #    feedinput[self.keep_prob_in]=feed_keepprob_save
                    icount += 1

                # Every 10 epochs, report the RMSE on the entire training set
                if icount_global % 10 == 0:
                    if self.miniBatch:
                        feedinput = self.generateFeedInput(range(len(keylist)),
                                                    energies,
                                                    atomArraysAll,
                                                    atomArraysAllDerivs,
                                                    nAtomsDict,
                                                    atomsIndsReverse,
                                                    len(keylist),
                                                    trainingrate,
                                                    1.,1.,
                                                    natoms,
                                                    forcesExp=forces,
                                                    energycoefficient=self.parameters['energy_coefficient'],
                                                    forcecoefficient=self.parameters['force_coefficient'])
                    feed_keepprob_save=feedinput[self.keep_prob_in]
                    feed_keepprob_save_input=feedinput[self.input_keep_prob_in]
                    feedinput[self.keep_prob_in]=1.
                    feedinput[self.input_keep_prob_in]=1.
                    if self.parameters['force_coefficient'] > 1.e-5:
                        converge_save.append([self.sess.run(self.loss,feed_dict=feedinput),
                                         self.sess.run(self.energy_loss,feed_dict=feedinput),
                                         self.sess.run(self.force_loss,feed_dict=feedinput),
                                         self.sess.run(self.energy_maxresid,feed_dict=feedinput),
                                         self.sess.run(self.force_maxresid,feed_dict=feedinput)])
                        if len(converge_save)>10:
                            converge_save.pop(0)
                        convergence_vals=np.mean(converge_save,0)
                        converged=lf.check_convergence(*convergence_vals)
                        if converged:
                            raise ConvergenceOccurred()
                    else:
                        converged=lf.check_convergence(self.sess.run(self.energy_loss,feed_dict=feedinput),
                                         self.sess.run(self.energy_loss,feed_dict=feedinput),
                                         0.,
                                         self.sess.run(self.energy_maxresid,feed_dict=feedinput),
                                         0.)
                        if converged:
                            raise ConvergenceOccurred()
                    feedinput[self.keep_prob_in]=keepprob
                    feedinput[self.input_keep_prob_in]=inputkeepprob
                icount_global += 1
            return

        def trainmodelBFGS(maxEpochs):
            curinds=range(len(keylist))
            feedinput = self.generateFeedInput(curinds,
                                                           energies,
                                                           atomArraysAll,
                                                           atomArraysAllDerivs,
                                                           nAtomsDict,
                                                           atomsIndsReverse,
                                                           batchsize,
                                                           1.,
                                                           1.,
                                                           1.,
                                                           natoms,
                                                           forcesExp=forces,
                                                           energycoefficient=self.parameters['energy_coefficient'],
                                                           forcecoefficient=self.parameters['force_coefficient'])

            def step_callbackfun_forces(x):
                converged=lf.check_convergence(varlist[0](x),varlist[1](x),varlist[2](x),varlist[3](x),varlist[4](x))
                if converged:
                    raise ConvergenceOccurred()
                    
            def step_callbackfun_noforces(x):
                converged=lf.check_convergence(varlist[1](x),varlist[1](x),0.,varlist[3](x),0.)
                if converged:
                    raise ConvergenceOccurred()
                
            if (self.parameters['force_coefficient'] < 1.e-5):
                step_callbackfun=step_callbackfun_noforces
                curloss=self.energy_loss
            else:
                step_callbackfun=step_callbackfun_forces
                curloss=self.loss
            extOpt=ScipyOptimizerInterface(curloss,method='l-BFGS-b',options={'maxiter':maxEpochs,'ftol':1.e-10,'gtol':1.e-10,'factr':1.e4})
            varlist=[]
            for var in [self.loss,self.energy_loss,self.force_loss,self.energy_maxresid,self.force_maxresid]:
                varlist.append(extOpt._make_eval_func(var, self.sess, feedinput, []))

            extOpt.minimize(self.sess,feed_dict=feedinput,step_callback=step_callbackfun)
                
            return 

        try:        
            if self.optimizationMethod=='l-BFGS-b':
                trainmodelBFGS(self.maxTrainingEpochs)
            elif self.optimizationMethod=='ADAM':
                trainmodel(self.initialTrainingRate,
                          self.keep_prob, self.input_keep_prob,self.maxTrainingEpochs)
            else:
                log('uknown optimizer!')
        except ConvergenceOccurred:
            return True
        return False

               
    def get_energy_list(self, hashs, fingerprintDB, fingerprintDerDB=None, keep_prob=1., input_keep_prob=1.,forces=False,nsamples=1):
        """Methods to get the energy and forces for a set of
        configurations."""

        # Make images a list in case we've been passed a single hash to
        # calculate.
        if not(isinstance(hashs, list)):
            hashs = [hashs]

        # Reformat the image and fingerprint data into something we can pass
        # into tensorflow.

        atomArraysAll, nAtomsDict, atomsIndsReverse, natoms, atomArraysAllDerivs = generateTensorFlowArrays(
            fingerprintDB, self.elements, hashs, fingerprintDerDB, self.maxAtomsForces)

        energies = np.zeros(len(hashs))
        curinds = range(len(hashs))
        atomArraysFinal, atomArraysDerivsFinal, atomInds = generateBatch(
            curinds, self.elements, atomArraysAll, nAtomsDict, atomsIndsReverse, atomArraysAllDerivs)
        feedinput = {}
        tilederivs = []
        for element in self.elements:
            if len(atomArraysFinal[element]) > 0:
                feedinput[self.tensordict[element]] = atomArraysFinal[
                    element] / self.parameters['elementFPScales'][element]
                feedinput[self.indsdict[element]] = atomInds[element]
                feedinput[self.maskdict[element]] = np.ones((len(hashs), 1))
                if forces:
                    feedinput[self.tensorDerivDict[element]
                              ] = atomArraysDerivsFinal[element]
                    if len(atomArraysDerivsFinal[element]) > 0:
                        tilederivs = np.array([1, atomArraysDerivsFinal[element].shape[
                                              1], atomArraysDerivsFinal[element].shape[2], 1])
            else:
                feedinput[self.tensordict[element]] = np.zeros(
                    (1, self.elementFingerprintLengths[element]))
                feedinput[self.indsdict[element]] = [0]
                feedinput[self.maskdict[element]] = np.zeros((len(hashs), 1))
                feedinput[self.tensorDerivDict[element]] = np.zeros(
                    (1, 1, 3, self.elementFingerprintLengths[element]))
        feedinput[self.batchsizeInput] = len(hashs)
        feedinput[self.nAtoms_in] = natoms[curinds]
        feedinput[self.keep_prob_in] = keep_prob
        feedinput[self.input_keep_prob_in] = input_keep_prob
        feedinput[self.energyProdScale]=self.parameters['energyProdScale']
        if tilederivs == []:
            tilederivs = [1, 1, 1, 1]
        feedinput[self.tileDerivs] = tilederivs
        if nsamples==1:
            energies = np.array(self.sess.run(self.energy,feed_dict=feedinput)) + self.parameters['energyMeanScale']
            # Add in the per-atom base energy.
            natomsArray = np.zeros((len(hashs), len(self.elements)))
            for i in range(len(hashs)):
                for j in range(len(self.elements)):
                    natomsArray[i][j] = nAtomsDict[self.elements[j]][i]
            if self.parameters['applyLinearModel']:
                energies = energies + self.linearmodel.predict(natomsArray)
        else:
            energysave=[]
            # Add in the per-atom base energy.
            natomsArray = np.zeros((len(hashs), len(self.elements)))
            for i in range(len(hashs)):
                for j in range(len(self.elements)):
                    natomsArray[i][j] = nAtomsDict[self.elements[j]][i]
            for samplenum in range(nsamples):
                energies = np.array(self.sess.run(self.energy,feed_dict=feedinput)) + self.parameters['energyMeanScale']
                if self.parameters['applyLinearModel']:
                    energies = energies + self.linearmodel.predict(natomsArray)
                energysave.append(map(lambda x: x[0],energies))
            energies=np.array(energysave)
        if forces:
            force = self.sess.run(self.forces,
                feed_dict=feedinput) 
        else:
            force = []
        return energies, force

    def get_energy(self, fingerprint):
        """Get the energy by feeding in a list to the get_list version (which
        is more efficient for anything greater than 1 image)."""
        key = '1'
        energies, forces = self.get_energy_list([key], {key: fingerprint})
        return energies[0]

    def getVariance(self,fingerprint,nSamples=10,l=1.):
        key = '1'
        #energies=[]
        #for i in range(nSamples):
        #    energies.append(self.get_energy_list([key], {key: fingerprint},keep_prob=self.keep_prob)[0])
        energies,force=self.get_energy_list([key], {key: fingerprint},keep_prob=self.keep_prob,nsamples=nSamples)
        if 'regularization_strength' in self.parameters and self.parameters['regularization_strength'] is not None:
            tau=l**2.*self.keep_prob/(2*self.parameters['numTrainingImages']*self.parameters['regularization_strength'])
            var=np.var(energies)+tau**-1.
        else:
            tau=1
            var=np.var(energies)
        return var

    def get_forces(self, fingerprint, derfingerprint):
    # get_forces function still needs to be implemented.  Can't do this
    # without the fingerprint derivates working properly though
        key = '1'
        energies, forces = self.get_energy_list(
            [key], {key: fingerprint}, fingerprintDerDB={key: derfingerprint}, forces=True)
        return forces[0][0:len(fingerprint)]

    def tostring(self):
        """Dummy tostring to make things work."""
        params = {}

        params['hiddenlayers'] = self.hiddenlayers
        params['keep_prob'] = self.keep_prob
        params['input_keep_prob'] = self.input_keep_prob
        params['elementFingerprintLengths'] = self.elementFingerprintLengths
        params['batchsize'] = self.batchsize
        params['maxTrainingEpochs'] = self.maxTrainingEpochs
        params['initialTrainingRate'] = self.initialTrainingRate
        params['activation'] = self.activationName
        params['saveVariableName'] = self.saveVariableName
        params['parameters'] = self.parameters
        params['miniBatch'] = self.miniBatch
        params['maxAtomsForces']=self.maxAtomsForces
        params['optimizationMethod']=self.optimizationMethod
        params['applyLinearModel']=self.parameters['applyLinearModel']

        # Create a string format of the tensorflow variables.
        self.saver.save(self.sess, 'tfAmpNN-checkpoint')
        with open('tfAmpNN-checkpoint') as fhandle:
            params['tfVars'] = fhandle.read()

        # Unfortunately, scikit learn only can use the pickle for
        # saving/reestablishing itself.
        params['scikit_model'] = pickle.dumps(self.linearmodel)

        return str(params)

def model(x, segmentinds, keep_prob, input_keep_prob,batchsize, neuronList, activationType,
          fplength, mask, name, dxdxik, tilederiv,element):
    """Generates a multilayer neural network with variable number
    of neurons, so that we have a template for each atom's NN."""
    namefun=lambda x: '%s_%s_'%(name,element)+x
    nNeurons = neuronList[0]
    # Pass  the input tensors through the first soft-plus layer
    W_fc = weight_variable([fplength, nNeurons], name=namefun('Wfc0'))
    b_fc = bias_variable([nNeurons], name=namefun('bfc0'))
    input_dropout=tf.nn.dropout(x,input_keep_prob)
    #h_fc = activationType(tf.matmul(x, W_fc) + b_fc)
    h_fc = tf.nn.dropout(activationType(tf.matmul(input_dropout, W_fc) + b_fc),keep_prob)
    #l2_regularization=tf.reduce_sum(tf.square(W_fc))+tf.reduce_sum(tf.square(b_fc))
    l2_regularization=tf.reduce_sum(tf.square(W_fc))
    if len(neuronList) > 1:
        for i in range(1, len(neuronList)):
            nNeurons = neuronList[i]
            nNeuronsOld = neuronList[i - 1]
            W_fc = weight_variable([nNeuronsOld, nNeurons], name=namefun('Wfc%d'%i))
            b_fc = bias_variable([nNeurons], name=namefun('bfc%d'%i))
            h_fc = tf.nn.dropout(activationType(
                tf.matmul(h_fc, W_fc) + b_fc), keep_prob)
            l2_regularization+=tf.reduce_sum(tf.square(W_fc))+tf.reduce_sum(tf.square(b_fc))
            #l2_regularization+=tf.reduce_sum(tf.square(W_fc))

    W_fc_out = weight_variable([neuronList[-1], 1], name=namefun('Wfcout'))
    b_fc_out = bias_variable([1], name=namefun('bfcout'))
    y_out = tf.matmul(h_fc, W_fc_out) + b_fc_out
    l2_regularization+=tf.reduce_sum(tf.square(W_fc_out))+tf.reduce_sum(tf.square(b_fc_out))
    #l2_regularization+=tf.reduce_sum(tf.square(W_fc_out)))

    # Sum the predicted energy for each molecule
    reducedSum = tf.unsorted_segment_sum(y_out, segmentinds, batchsize)

    dEjdgj = tf.gradients(y_out, x)[0]
    dEjdgj1 = tf.expand_dims(dEjdgj, 1)
    dEjdgj2 = tf.expand_dims(dEjdgj1, 1)
    dEjdgjtile = tf.tile(dEjdgj2, tilederiv)
    dEdxik = tf.mul(dxdxik, dEjdgjtile)
    dEdxikReduce = tf.reduce_sum(dEdxik, 3)
    dEdxik_reduced = tf.unsorted_segment_sum(
        dEdxikReduce, segmentinds, batchsize)
    return tf.mul(reducedSum, mask), dEdxik_reduced,l2_regularization


def weight_variable(shape, name, stddev=0.1):
    """Helper functions taken from the MNIST tutorial to generate weight and
    bias variables with random initial weights."""
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name, a=0.1):
    """Helper functions taken from the MNIST tutorial to generate weight and
    bias variables with random initial weights."""
    initial = tf.truncated_normal(stddev=a, shape=shape)
    return tf.Variable(initial, name=name)



def generateBatch(curinds, elements, atomArraysAll, nAtomsDict,
                  atomsIndsReverse, atomArraysAllDerivs):
    """This method generates batches from a large dataset using a set of
    selected indices curinds."""
    # inputs:
    atomArraysFinal = {}
    atomArraysDerivsFinal = {}
    for element in elements:
        validKeys = np.in1d(atomsIndsReverse[element], curinds)
        if len(validKeys) > 0:
            atomArraysFinal[element] = atomArraysAll[element][validKeys]
            if len(atomArraysAllDerivs[element]) > 0:
                atomArraysDerivsFinal[element] = atomArraysAllDerivs[
                    element][validKeys, :, :, :]
            else:
                atomArraysDerivsFinal[element] = []
        else:
            atomArraysFinal[element] = []
            atomArraysDerivsFinal[element] = []

    atomInds = {}
    for element in elements:
        validKeys = np.in1d(atomsIndsReverse[element], curinds)
        if len(validKeys) > 0:
            atomIndsTemp = np.sum(atomsIndsReverse[element][validKeys], 1)
            atomInds[element] = atomIndsTemp * 0.
            for i in range(len(curinds)):
                atomInds[element][atomIndsTemp == curinds[i]] = i
        else:
            atomInds[element] = []

    return atomArraysFinal, atomArraysDerivsFinal, atomInds


def generateTensorFlowArrays(fingerprintDB, elements, keylist,
                             fingerprintDerDB=None, maxAtomsForces=0):
    """
    This function generates the inputs to the tensorflow graph for the selected images.
    The essential problem is that each neural network is associated with a
    specific element type.  Thus, atoms in each ASE image need to be sent to
    different networks.

    Inputs:

    fingerprintDB: a database of fingerprints, as taken from the descriptor

    elements: a list of element types (e.g. 'C','O', etc)

    keylist: a list of hashs into the fingerprintDB that we want to creat inputs for

    fingerprintDerDB: a database of fingerprint derivatives, as taken from the descriptor

    maxAtomsForces: the maximum length of the atoms

    Outputs:

    atomArraysAll: a dictionary of fingerprint inputs to each element's neural
        network

    nAtomsDict: a dictionary for each element with lists of the number of
        atoms of each type in each image

    atomsIndsReverse: a dictionary that contains the index of each atom into
        the original keylist

    nAtoms: the number of atoms in each image

    atomArraysAllDerivs: dictionary of fingerprint derivates for each
        element's neural network
    """

    nAtomsDict = {}
    for element in elements:
        nAtomsDict[element] = np.zeros(len(keylist))

    for j in range(len(keylist)):
        fp = fingerprintDB[keylist[j]]
        atomSymbols, fpdata = zip(*fp)
        for i in range(len(fp)):
            nAtomsDict[atomSymbols[i]][j] += 1

    atomsPositions = {}
    for element in elements:
        atomsPositions[element] = np.cumsum(
            nAtomsDict[element]) - nAtomsDict[element]

    atomsIndsReverse = {}
    for element in elements:
        atomsIndsReverse[element] = []
        for i in range(len(keylist)):
            if nAtomsDict[element][i] > 0:
                atomsIndsReverse[element].append(
                    np.ones((nAtomsDict[element][i], 1)) * i)
        if len(atomsIndsReverse[element]) > 0:
            atomsIndsReverse[element] = np.concatenate(
                atomsIndsReverse[element])

    atomArraysAll = {}
    for element in elements:
        atomArraysAll[element] = []

    natoms = np.zeros((len(keylist), 1))
    for j in range(len(keylist)):
        fp = fingerprintDB[keylist[j]]
        atomSymbols, fpdata = zip(*fp)
        atomdata = zip(atomSymbols, range(len(atomSymbols)))
        for element in elements:
            atomArraysTemp = []
            curatoms = [atom for atom in atomdata if atom[0] == element]
            for i in range(len(curatoms)):
                atomArraysTemp.append(fp[curatoms[i][1]][1])
            if len(atomArraysTemp) > 0:
                atomArraysAll[element].append(atomArraysTemp)
        natoms[j] = len(atomSymbols)

    for element in elements:
        if len(atomArraysAll[element]) > 0:
            atomArraysAll[element] = np.concatenate(atomArraysAll[element])
        else:
            atomArraysAll[element] = []

    # Set up the array for atom-based fingerprint derivatives.
    atomArraysAllDerivs = {}
    for element in elements:
        atomArraysAllDerivs[element] = []
    if fingerprintDerDB is not None:
        for j in range(len(keylist)):
            fp = fingerprintDB[keylist[j]]
            fpDer = fingerprintDerDB[keylist[j]]
            atomSymbols, fpdata = zip(*fp)
            atomdata = zip(atomSymbols, range(len(atomSymbols)))
            for element in elements:
                curatoms = [atom for atom in atomdata if atom[0] == element]
                if len(curatoms) > 0:
                    if maxAtomsForces == 0:
                        maxAtomsForces = len(atomdata)
                    fingerprintDerivatives = np.zeros(
                        (len(curatoms), maxAtomsForces, 3, len(fp[curatoms[0][1]][1])))
                    for i in range(len(curatoms)):
                        for k in range(len(atomdata)):
                            for ix in range(3):
                                dictkey = (k, atomdata[k][0], curatoms[
                                           i][1], curatoms[i][0], ix)
                                if dictkey in fpDer:
                                    fingerprintDerivatives[
                                        i, k, ix, :] = fpDer[dictkey]
                    atomArraysAllDerivs[element].append(fingerprintDerivatives)
        for element in elements:
            if len(atomArraysAllDerivs[element]) > 0:
                atomArraysAllDerivs[element] = np.concatenate(
                    atomArraysAllDerivs[element], axis=0)
            else:
                atomArraysAllDerivs[element] = []

    return atomArraysAll, nAtomsDict, atomsIndsReverse, natoms, atomArraysAllDerivs
