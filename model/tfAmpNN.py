import numpy as np
import tensorflow as tf
import random
import string

class tfAmpNN:
    def __init__(self, elementFingerprintLengths, hiddenlayers=[5, 5], activation='relu', keep_prob=0.5, RMSEtarget=1e-2, maxTrainingEpochs=10000,batchsize=20,initialTrainingRate=1e-4,miniBatch=True,tfVars=None,saveVariableName=None,parameters={},sess=None):
        # We expect a diction elementFingerprintLengths, which maps elements to
        # the lengths of their corresponding fingerprints
        self.hiddenlayers = hiddenlayers
        if isinstance(activation, basestring):
            self.activationName=activation
            self.activation = eval('tf.nn.'+activation)
        else:
            self.activation=activation
            self.activationName=activation.__name__
        self.keep_prob = keep_prob
        self.elements = elementFingerprintLengths.keys()
        if saveVariableName is None:
            self.saveVariableName=''.join(random.choice(string.ascii_uppercase + string.digits+string.ascii_lowercase) for _ in range(10))
        else:
            self.saveVariableName=saveVariableName
        self.elementFingerprintLengths = elementFingerprintLengths
        self.constructModel()
        if sess is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess=sess
        self.saver=tf.train.Saver(tf.trainable_variables())

        if tfVars is not None:
            trainable_vars=tf.trainable_variables()
            all_vars=tf.all_variables()
            untrainable_vars=[]
            for var in all_vars:
                if var not in trainable_vars:
                    untrainable_vars.append(var)
            self.sess.run(tf.initialize_variables(untrainable_vars))
            with open('tfAmpNN-checkpoint-restore','w') as fhandle:
                fhandle.write(tfVars)
            self.saver.restore(self.sess,'tfAmpNN-checkpoint-restore')
        else:
            self.initializeVariables()
        self.RMSEtarget = RMSEtarget
        self.maxTrainingEpochs = maxTrainingEpochs
        self.batchsize = batchsize
        self.initialTrainingRate=initialTrainingRate
        self.miniBatch=miniBatch
        self.parameters=parameters
        for prop in ['elementFPScales','energyMeanScale','energyProdScale','energyPerElement']:
            if prop not in parameters:
                self.parameters[prop]=None

    def constructModel(self):
        # first, define all of the input placeholders to the tf system
        tensordict = {}
        indsdict = {}
        maskdict = {}
        tensorDerivDict={}
        for element in self.elements:
            tensordict[element] = tf.placeholder("float", shape=[None, self.elementFingerprintLengths[element]])
            tensorDerivDict[element]=tf.placeholder("float", shape=[None,None,3,self.elementFingerprintLengths[element]])
            indsdict[element] = tf.placeholder("int64", shape=[None])
            maskdict[element] = tf.placeholder("float", shape=[None,1])
        self.indsdict = indsdict
        self.tileDerivs=tf.placeholder("int32",shape=[4])
        self.tensordict = tensordict
        self.maskdict = maskdict
        self.tensorDerivDict=tensorDerivDict
        # y_ is the known energy of each configuration
        self.y_ = tf.placeholder("float", shape=[None, 1])
        self.keep_prob_in = tf.placeholder("float")
        self.nAtoms_in = tf.placeholder("float", shape=[None, 1])
        self.batchsizeInput = tf.placeholder("int32")
        self.learningrate = tf.placeholder("float")
        self.forces_in = tf.placeholder("float",shape=[None,None,3])
        self.energycoefficient=tf.placeholder("float")
        self.forcecoefficient=tf.placeholder("float")

        # generate a multilayer neural network for each element type
        outdict = {}
        forcedict={}
        for element in self.elements:
            if isinstance(self.hiddenlayers, dict):
                networkListToUse = self.hiddenlayers[element]
            else:
                networkListToUse = self.hiddenlayers
            outdict[element],forcedict[element] = model(tensordict[element], indsdict[element], self.keep_prob_in, self.batchsizeInput,
                                     networkListToUse, self.activation, self.elementFingerprintLengths[element], mask=maskdict[element],name=self.saveVariableName,dxdxik=self.tensorDerivDict[element],tilederiv=self.tileDerivs)
        self.outdict = outdict

        # The total energy is the sum of the energies over each atom type
        keylist = self.elements
        ytot = outdict[keylist[0]]
        for i in range(1, len(keylist)):
            ytot = ytot + outdict[keylist[i]]
        self.energy = ytot

        # The total energy is the sum of the energies over each atom type
        Ftot = forcedict[keylist[0]]
        for i in range(1, len(keylist)):
            Ftot = Ftot + forcedict[keylist[i]]
        self.forces=-Ftot
        
        # Define output nodes for the energy of a configuration, a loss
        # function, and the loss per atom (which is what we usually track)
        self.loss = tf.sqrt(tf.reduce_mean(
            tf.square(tf.sub(self.energy, self.y_))))
        self.lossPerAtom = tf.sqrt(tf.reduce_mean(
            tf.square(tf.div(tf.sub(self.energy, self.y_), self.nAtoms_in))))

        self.train_step = tf.train.AdamOptimizer(
            self.learningrate,beta1=0.9).minimize(self.loss)

        self.loss_forces=self.energycoefficient*tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.energy, self.y_))))+self.forcecoefficient*tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.forces_in, self.forces))))
        self.train_step_forces = tf.train.AdamOptimizer(self.learningrate,beta1=0.9).minimize(self.loss_forces)
    # this function resets all of the variables in the current tensorflow model
    def initializeVariables(self):
        self.sess.run(tf.initialize_all_variables())

    def generateFeedInput(self, curinds, energies, atomArraysAll,atomArraysAllDerivs, nAtomsDict, atomsIndsReverse, batchsize, trainingrate, keepprob, natoms,forcesExp=0.,forces=False,energycoefficient=1.,forcecoefficient=None):
        atomArraysFinal, atomArraysDerivsFinal,atomInds = generateBatch(
            curinds, self.elements, atomArraysAll, nAtomsDict, atomsIndsReverse,atomArraysAllDerivs)
        feedinput = {}
        for element in self.elements:
            if len(atomArraysFinal[element]) > 0:
                feedinput[self.tensordict[element]] = atomArraysFinal[
                    element] / self.parameters['elementFPScales'][element]
                feedinput[self.indsdict[element]] = atomInds[element]
                feedinput[self.maskdict[element]] = np.ones((batchsize,1))
                feedinput[self.tensorDerivDict[element]]=atomArraysDerivsFinal[element]
                if len(atomArraysDerivsFinal[element])>0:
                    tilederivs=np.array([1,atomArraysDerivsFinal[element].shape[1],atomArraysDerivsFinal[element].shape[2],1])
            else:
                feedinput[self.tensordict[element]] = np.zeros(
                    (1, self.elementFingerprintLengths[element]))
                feedinput[self.indsdict[element]] = [0]
                feedinput[self.maskdict[element]] = np.zeros((batchsize,1))
        feedinput[self.tileDerivs]=tilederivs
        feedinput[self.y_] = energies[curinds]
        feedinput[self.batchsizeInput] = batchsize
        feedinput[self.learningrate] = trainingrate
        feedinput[self.keep_prob_in] = keepprob
        feedinput[self.nAtoms_in] = natoms[curinds]
        if forcecoefficient is not None:
            feedinput[self.forces_in]=forcesExp
            feedinput[self.forcecoefficient]=forcecoefficient
            feedinput[self.energycoefficient]=energycoefficient
        return feedinput

    def fit(self, trainingimages, descriptor, cores=1, log=[],energy_coefficient=1.,force_coefficient=None):
        batchsize = self.batchsize
        if force_coefficient is None:
            log('Training the Tensorflow network!')
        else:
            log('Training the Tensorflow network w/ Forces!')
        images = trainingimages
        fingerprintDB = descriptor.fingerprints
        fingerprintDerDB=descriptor.derfingerprints
        atomArraysAll, nAtomsDict, atomsIndsReverse, natoms,atomArraysAllDerivs = generateTensorFlowArrays(
             fingerprintDB, self.elements, images.keys(),fingerprintDerDB)
        energies=map(lambda x: [images[x].get_potential_energy()],images.keys())
        energies=np.array(energies)
        
        natomsArray = np.zeros((len(images), len(self.elements)))
        for i in range(len(images)):
            for j in range(len(self.elements)):
                natomsArray[i][j] = nAtomsDict[self.elements[j]][i]
        if np.linalg.matrix_rank(natomsArray)==len(self.elements):
            energyPerElement = np.linalg.lstsq(a=natomsArray, b=energies)
            energyPerElement = energyPerElement[0]
        else:
            energyPerElement=np.zeros((len(self.elements),1))
        self.parameters['energyPerElement'] = {}
        for j in range(len(self.elements)):
            self.parameters['energyPerElement'][self.elements[j]] = energyPerElement[j]
        energies = energies - np.dot(natomsArray, energyPerElement)

        self.parameters['energyMeanScale'] = np.mean(energies)
        #self.parameters['energyMeanScale'] = 0.
        energies = energies - self.parameters['energyMeanScale']
        self.parameters['energyProdScale'] = np.mean(np.abs(energies))
        self.parameters['energyProdScale'] = 1.
        energies = energies / self.parameters['energyProdScale']
        self.parameters['elementFPScales'] = {}
        for element in self.elements:
            if len(atomArraysAll[element]) == 0:
                self.parameters['elementFPScales'] = 1.
            else:
                self.parameters['elementFPScales'][element] = np.max(
                    np.max(atomArraysAll[element]))
                self.parameters['elementFPScales'][element]=1.

        if force_coefficient is not None:
            forces=map(lambda x: images[x].get_forces(apply_constraint=False),images.keys())
        else:
            forces=0.

        if not(self.miniBatch):
            batchsize = len(images)

        def trainmodel(targetRMSE, trainingrate, keepprob, maxepochs):
            icount = 1
            icount_global=1
            indlist = np.arange(len(images))
            RMSE = targetRMSE + 1.
            while (RMSE > targetRMSE) & (icount < maxepochs):

                if self.miniBatch:
                    np.random.shuffle(indlist)

                for i in range(int(len(images) / batchsize)):
                    # For each batch, construct a new set of inputs
                    if self.miniBatch or (not(self.miniBatch)and(icount == 1)):
                        if self.miniBatch:
                            curinds = indlist[
                                np.arange(batchsize) + i * batchsize]
                        else:
                            curinds = range(len(images))

                        feedinput = self.generateFeedInput(
                            curinds, energies, atomArraysAll, atomArraysAllDerivs,nAtomsDict, atomsIndsReverse, batchsize, trainingrate, keepprob, natoms,forcesExp=forces,energycoefficient=energy_coefficient,forcecoefficient=force_coefficient)

                    # run a training step with the new inputs
                    if force_coefficient is None:
                        self.sess.run(self.train_step, feed_dict=feedinput)
                    else:
                        self.sess.run(self.train_step_forces, feed_dict=feedinput)

                    # Print the loss function every 100 evals.  Would be better
                    # to handle this by evaluating the loss function on the
                    # entire dataset, but batchsize is currently hardcoded at
                    # the moment
                    if icount % 100 == 0:
                        log('batch RMSE(energy)=%1.3e, # Epochs=%d' % (self.loss.eval(feed_dict=feedinput) * self.parameters['energyProdScale'],icount))
                    icount+=1
                
                if icount_global% 10==0:
                    RMSE = self.loss.eval(feed_dict=self.generateFeedInput(range(len(images)), energies, atomArraysAll,atomArraysAllDerivs,
                                                                       nAtomsDict, atomsIndsReverse, len(images), trainingrate, keepprob, natoms,forcesExp=forces,energycoefficient=energy_coefficient,forcecoefficient=force_coefficient)) * self.parameters['energyProdScale']
                    log('global RMSE=%1.3f'%(RMSE))
                    if force_coefficient is not None:
                        RMSE_combined=self.loss_forces.eval(feed_dict=self.generateFeedInput(range(len(images)), energies, atomArraysAll,atomArraysAllDerivs,
                                                                       nAtomsDict, atomsIndsReverse, len(images), trainingrate, keepprob, natoms,forcesExp=forces,energycoefficient=energy_coefficient,forcecoefficient=force_coefficient))
                        log('combined loss function (energy+force)=%1.3f'%(RMSE_combined))
                    #if icount_global % 100==0:
                    #    log('global RMSE=%1.3f'%(RMSE))
                icount_global+=1
            return RMSE

        RMSE = trainmodel(self.RMSEtarget, self.initialTrainingRate,
                          0.5, self.maxTrainingEpochs)
        if RMSE < self.RMSEtarget:
            return True
        else:
            return False

    # implement methods to get the energy and forces for a set of
    # configurations
    def get_energy_list(self, hashs, fingerprintDB,fingerprintDerDB=None,keep_prob=1.,forces=False):

        # make images a list in case we've been passed a single hash to
        # calculate
        if not(isinstance(hashs, list)):
            hashs = [hashs]

        # reformat the image and fingerprint data into something we can pass
        # into tensorflow
        atomArraysAll, nAtomsDict, atomsIndsReverse, natoms, atomArraysAllDerivs = generateTensorFlowArrays(
             fingerprintDB, self.elements, hashs,fingerprintDerDB)

        energies = np.zeros(len(hashs))
        curinds = range(len(hashs))
        atomArraysFinal, atomArraysDerivsFinal,atomInds = generateBatch(
            curinds, self.elements, atomArraysAll, nAtomsDict, atomsIndsReverse,atomArraysAllDerivs)
        feedinput = {}
        tilederivs=[]
        for element in self.elements:
            if len(atomArraysFinal[element]) > 0:
                feedinput[self.tensordict[element]] = atomArraysFinal[element] / self.parameters['elementFPScales'][element]
                feedinput[self.indsdict[element]] = atomInds[element]
                feedinput[self.maskdict[element]] = np.ones((len(hashs),1))
                if forces:
                    feedinput[self.tensorDerivDict[element]]=atomArraysDerivsFinal[element]
                    if len(atomArraysDerivsFinal[element])>0:
                        tilederivs=np.array([1,atomArraysDerivsFinal[element].shape[1],atomArraysDerivsFinal[element].shape[2],1])
            else:
                feedinput[self.tensordict[element]] = np.zeros(
                    (1, self.elementFingerprintLengths[element]))
                feedinput[self.indsdict[element]] = [0]
                feedinput[self.maskdict[element]] = np.zeros((len(hashs),1))
        feedinput[self.batchsizeInput] = len(hashs)
        feedinput[self.nAtoms_in] = natoms[curinds]
        feedinput[self.keep_prob_in]=keep_prob
        if tilederivs==[]:
            tilederivs=[1,1,1,1]
        feedinput[self.tileDerivs]=tilederivs
        energies = np.array(self.energy.eval(feed_dict=feedinput)) * self.parameters['energyProdScale'] + self.parameters['energyMeanScale']
        
        #Add in the per-atom base energy
        natomsArray = np.zeros((len(hashs), len(self.elements)))
        for i in range(len(hashs)):
            for j in range(len(self.elements)):
                natomsArray[i][j] = nAtomsDict[self.elements[j]][i]
        energyPerElement=np.zeros(len(self.elements))
        for j in range(len(self.elements)):
            energyPerElement[j]=self.parameters['energyPerElement'][self.elements[j]]

        energies = energies + np.reshape(np.dot(natomsArray, energyPerElement),(len(energies),1))

        if forces:
            force=self.forces.eval(feed_dict=feedinput) * self.parameters['energyProdScale']
        else:
            force=[]
        return energies,force

    #Get the energy by feeding in a list to the get_list version (which is more efficient for anything greater than 1 image)
    def get_energy(self,fingerprint):
        key='1'
        energies,forces=self.get_energy_list([key],{key: fingerprint})
        return energies[0]

    # get_forces function still needs to be implemented.  Can't do this without the fingerprint derivates working properly though
    def get_forces(self,fingerprint,derfingerprint):
        key='1'
        energies,forces=self.get_energy_list([key],{key: fingerprint},fingerprintDerDB={key: derfingerprint},forces=True)
        return forces[0]
    
    #Dummy tostring to make things work
    def tostring(self):
        params={}
        
        params['hiddenlayers']=self.hiddenlayers
        params['keep_prob']=self.keep_prob
        params['elementFingerprintLengths']=self.elementFingerprintLengths
        params['RMSEtarget']=self.RMSEtarget
        params['batchsize']=self.batchsize
        params['maxTrainingEpochs']=self.maxTrainingEpochs
        params['initialTrainingRate']=self.initialTrainingRate
        params['activation']=self.activationName
        params['saveVariableName']=self.saveVariableName
        params['parameters']=self.parameters

        params['miniBatch']=self.miniBatch
        
        self.saver.save(self.sess,'tfAmpNN-checkpoint')
        with open('tfAmpNN-checkpoint') as fhandle:
            params['tfVars']=fhandle.read()
        #params['tfVars']='tfAmpNN-checkpoint'
        return str(params)
        #return open()


# This function generates a multilayer neural network with variable number
# of neurons, so that we have a template for each atom's NN
def model(x, segmentinds, keep_prob, batchsize, neuronList, activationType, fplength, mask,name,dxdxik,tilederiv):

    nNeurons = neuronList[0]
    # Pass  the input tensors through the first soft-plus layer
    W_fc = weight_variable([fplength, nNeurons],name=name)
    b_fc = bias_variable([nNeurons],name=name)
    h_fc = tf.nn.dropout(activationType(tf.matmul(x, W_fc) + b_fc),keep_prob)

    if len(neuronList) > 1:
        for i in range(1, len(neuronList)):
            nNeurons = neuronList[i]
            nNeuronsOld = neuronList[i - 1]
            W_fc = weight_variable([nNeuronsOld, nNeurons],name=name)
            b_fc = bias_variable([nNeurons],name=name)
            h_fc = tf.nn.dropout(activationType(tf.matmul(h_fc, W_fc) + b_fc),keep_prob)

    W_fc_out = weight_variable([neuronList[-1], 1],name=name)
    b_fc_out = bias_variable([1], a=-1e-5,name=name)
    y_out = tf.matmul(h_fc, W_fc_out) + b_fc_out

    # Sum the predicted energy for each molecule
    reducedSum = tf.unsorted_segment_sum(y_out, segmentinds, batchsize)

    dEjdgj=tf.gradients(y_out,x)[0]
    dEjdgj1=tf.expand_dims(dEjdgj,1)
    dEjdgj2=tf.expand_dims(dEjdgj1,1)
    dEjdgjtile=tf.tile(dEjdgj2,tilederiv)
    dEdxik=tf.mul(dxdxik,dEjdgjtile)
    dEdxikReduce=tf.reduce_sum(dEdxik,3)
    dEdxik_reduced = tf.unsorted_segment_sum(dEdxikReduce, segmentinds, batchsize)
    return tf.mul(reducedSum, mask),dEdxik_reduced
    #return reducedSum


# Helper functions taken from the MNIST tutorial to generate weight and
# bias variables
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=1.)
    return tf.Variable(initial,name=name)

def bias_variable(shape, name,a=0.5):
    initial = tf.constant(a, shape=shape)
    return tf.Variable(initial,name=name)

# This method generates batches from a large dataset using a set of
# selected indices curinds.
def generateBatch(curinds, elements, atomArraysAll, nAtomsDict, atomsIndsReverse,atomArraysAllDerivs):
    atomArraysFinal = {}
    atomArraysDerivsFinal = {}
    for element in elements:
        validKeys = np.in1d(atomsIndsReverse[element], curinds)
        if len(validKeys) > 0:
            atomArraysFinal[element] = atomArraysAll[element][validKeys]
            if len(atomArraysAllDerivs[element])>0:
                atomArraysDerivsFinal[element] = atomArraysAllDerivs[element][validKeys,:,:,:]
            else:
                atomArraysDerivsFinal[element]=[]
        else:
            atomArraysFinal[element] = []
            atomArraysDerivsFinal[element]=[]

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

    return atomArraysFinal, atomArraysDerivsFinal,atomInds


#This function generates the inputs to the tensorflow graph for the selected images
def generateTensorFlowArrays(fingerprintDB, elements, keylist,fingerprintDerDB=None):
    nAtomsDict = {}
    for element in elements:
        nAtomsDict[element] = np.zeros(len(keylist))

    for j in range(len(keylist)):
        fp = fingerprintDB[keylist[j]]
        atomSymbols,fpdata=zip(*fp)
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
        atomSymbols,fpdata=zip(*fp)
        atomdata=zip(atomSymbols,range(len(atomSymbols)))
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

    atomArraysAllDerivs = {}
    for element in elements:
        atomArraysAllDerivs[element] = []
    if fingerprintDerDB is not None:
        for j in range(len(keylist)):
            fp = fingerprintDB[keylist[j]]
            fpDer = fingerprintDerDB[keylist[j]]
            atomSymbols,fpdata=zip(*fp)
            atomdata=zip(atomSymbols,range(len(atomSymbols)))
            for element in elements:
                curatoms = [atom for atom in atomdata if atom[0] == element]
                if len(curatoms)>0:
                    fingerprintDerivatives=np.zeros((len(curatoms),len(atomdata),3,len(fp[curatoms[0][1]][1])))
                    for i in range(len(curatoms)):
                        for k in range(len(atomdata)):
                            for ix in range(3):
                                #dictkey=(curatoms[i][1],curatoms[i][0],k,atomdata[k][0],ix)
                                dictkey=(k,atomdata[k][0],curatoms[i][1],curatoms[i][0],ix)
                                if dictkey in fpDer:
                                    fingerprintDerivatives[i,k,ix,:]=fpDer[dictkey]
                    atomArraysAllDerivs[element].append(fingerprintDerivatives)
        for element in elements:
            if len(atomArraysAllDerivs[element]) > 0:
                atomArraysAllDerivs[element] = np.concatenate(atomArraysAllDerivs[element],axis=0)
            else:
                atomArraysAllDerivs[element] = []
        
    return atomArraysAll, nAtomsDict, atomsIndsReverse, natoms,atomArraysAllDerivs







