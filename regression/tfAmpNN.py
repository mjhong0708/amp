import ase
import numpy as np
import tensorflow as tf


class tfAmpNN:
    def __init__(self,elementFingerprintLengths,hiddenlayers=[5,5],activation=tf.nn.relu,keep_prob=0.):
        #We expect a diction elementFingerprintLengths, which maps elements to the lengths of their corresponding fingerprints
        self.hiddenlayers=hiddenlayers
        self.activation=activation
        self.keep_prob=keep_prob
        self.elements=elementFingerprintLengths.keys()
        self.elementFingerprintLengths=elementFingerprintLengths
        self.constructModel()
        self.sess = tf.InteractiveSession()
        self.initializeVariables()
    
    def constructModel(self):
        
        #first, define all of the input placeholders to the tf system
        tensordict={}
        indsdict={}
        maskdict={}
        for element in self.elements:
            tensordict[element]=tf.placeholder("float", shape=[None, self.elementFingerprintLengths[element]])
            indsdict[element]=tf.placeholder("int64", shape=[None])
            maskdict[element]=tf.placeholder("float",shape=[None])
        self.indsdict=indsdict
        self.tensordict=tensordict
        self.maskdict=maskdict
        #y_ is the known energy of each configuration
        self.y_ = tf.placeholder("float", shape=[None, 1])
        self.keep_prob = tf.placeholder("float")
        self.nAtoms_in=tf.placeholder("float",shape=[None,1])
        self.batchsizeInput=tf.placeholder("int32")
        self.learningrate=tf.placeholder("float")

        #generate a multilayer neural network for each element type
        outdict={}
        for element in self.elements:
            if isinstance(self.hiddenlayers,dict):
                networkListToUse=self.hiddenlayers[element]
            else:
                networkListToUse=self.hiddenlayers
            outdict[element]=model(tensordict[element],indsdict[element],self.keep_prob,self.batchsizeInput,networkListToUse,self.activation,self.elementFingerprintLengths[element],mask=maskdict[element])
        self.outdict=outdict

        #The total energy is the sum of the energies over each atom type
        keylist=self.elements
        ytot=outdict[keylist[0]]
        for i in range(1,len(keylist)):
            energy=ytot+outdict[keylist[i]]
        
        #Define output nodes for the energy of a configuration, a loss function, and the loss per atom (which is what we usually track)
        self.energy=energy
        self.loss=tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.energy,self.y_))))
        self.lossPerAtom=tf.sqrt(tf.reduce_mean(tf.square(tf.div(tf.sub(self.energy,self.y_),self.nAtoms_in))))

        self.train_step=tf.train.AdamOptimizer(self.learningrate).minimize(self.loss)

    #this function resets all of the variables in the current tensorflow model
    def initializeVariables(self):
        self.sess.run(tf.initialize_all_variables())

    def train(self,images,fingerprintDB,batchsize=20,nEpochs=100):

        atomArraysAll,nAtomsDict,atomsIndsReverse,energies,natoms=generateTensorFlowArrays(images,fingerprintDB,self.elements,images.keys())
        
        #energies=energies-np.mean(energies)
        #energies=energies
        self.energyMeanScale=np.mean(energies)
        energies=energies-self.energyMeanScale
        self.energyProdScale=np.mean(np.abs(energies))
        energies=energies/self.energyProdScale
        self.elementFPScales={}
        for element in self.elements:
            self.elementFPScales[element]=np.max(np.max(atomArraysAll[element]))

        def trainmodel(nepoch,trainingrate,keepprob):
            icount=1
            indlist=np.arange(len(images))
            for j in range(nepoch):
                np.random.shuffle(indlist)
                
                for i in range(int(len(images)/batchsize)):
                    #For each batch, construct a new set of inputs
                    curinds=indlist[np.arange(batchsize)+i*batchsize]

                    atomArraysFinal,atomInds=generateBatch(curinds,self.elements,atomArraysAll,nAtomsDict,atomsIndsReverse)
                    feedinput={}
                    for element in self.elements:
                        if len(atomArraysFinal[element])>0:
                            feedinput[self.tensordict[element]]=atomArraysFinal[element]/self.elementFPScales[element]
                            feedinput[self.indsdict[element]]=atomInds[element]
                            feedinput[self.maskdict[element]]=np.ones(batchsize)
                        else:
                            feedinput[self.tensordict[element]]=np.zeros((1,self.elementFingerprintLengths[element]))
                            feedinput[self.indsdict[element]]=[0]
                            feedinput[self.maskdict[element]]=np.zeros(batchsize)
                    feedinput[self.y_]=energies[curinds]
                    feedinput[self.batchsizeInput]=batchsize
                    feedinput[self.learningrate]=trainingrate
                    feedinput[self.keep_prob]=keepprob
                    feedinput[self.nAtoms_in]=natoms[curinds]

                    #run a training step with the new inputs
                    self.sess.run(self.train_step, feed_dict=feedinput)
                    
                    #Print the loss function every 100 evals.  Would be better to handle this by evaluating the loss function on the entire dataset, but batchsize is currently hardcoded at the moment
                    if icount%100==0:
                        print('Loss per atom=%1.3f'%(self.lossPerAtom.eval(feed_dict=feedinput)*self.energyProdScale))
                    icount+=1

        trainmodel(nEpochs,1.e-4,0.5)

    #implement methods to get the energy and forces for a set of configurations
    def get_energy(self,hashs,images,fingerprintDB):
        
        #make images a list in case we've been passed a single hash to calculate
        if not(isinstance(hashs,list)):
            hashs=[hashs]
        
        #reformat the image and fingerprint data into something we can pass into tensorflow
        atomArraysAll,nAtomsDict,atomsIndsReverse,energies,natoms=generateTensorFlowArrays(images,fingerprintDB,self.elements,hashs)
        
        energies=np.zeros(len(images))
        curinds=range(len(hashs))
        atomArraysFinal,atomInds=generateBatch(curinds,self.elements,atomArraysAll,nAtomsDict,atomsIndsReverse)
        feedinput={}
        for element in self.elements:
            if len(atomArraysFinal[element])>0:
                feedinput[self.tensordict[element]]=atomArraysFinal[element]/self.elementFPScales[element]
                feedinput[self.indsdict[element]]=atomInds[element]
                feedinput[self.maskdict[element]]=np.ones(len(hashs))
            else:
                feedinput[self.tensordict[element]]=np.zeros((1,self.elementFingerprintLengths[element]))
                feedinput[self.indsdict[element]]=[0]
                feedinput[self.maskdict[element]]=np.zeros(len(hashs))
        feedinput[self.batchsizeInput]=len(hashs)
        feedinput[self.nAtoms_in]=natoms[curinds]
        energies=np.array(self.energy.eval(feed_dict=feedinput))*self.energyProdScale+self.energyMeanScale
        return energies

    #def get_forces():


#This function generates a multilayer neural network with variable number of neurons
def model(x,segmentinds,keep_prob,batchsize,neuronList,activationType,fplength,mask):
    
    nNeurons=neuronList[0]
    #Pass  the input tensors through the first soft-plus layer
    W_fc1 = weight_variable([fplength, nNeurons])
    b_fc1 = bias_variable([nNeurons])
    h_fc1 = activationType(tf.matmul(x, W_fc1) + b_fc1)
    
    if len(neuronList)>1:
        for i in range(1,len(neuronList)):
            nNeurons=neuronList[i]
            nNeuronsOld=neuronList[i-1]
            W_fc = weight_variable([nNeuronsOld, nNeurons])
            b_fc = bias_variable([nNeurons])
            h_fc = activationType(tf.matmul(h_fc1, W_fc) + b_fc)

    W_fc_out = weight_variable([ neuronList[-1], 1])
    b_fc_out = bias_variable([1])
    y_out=tf.matmul(h_fc, W_fc_out) + b_fc_out
    
    #Sum the predicted energy for each atom
    reducedSum=tf.unsorted_segment_sum(y_out,segmentinds,batchsize)
    return tf.mul(reducedSum,mask)


#Helper functions taken from the MNIST tutorial to generate weight and bias variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)

#This method generates batches from a large dataset using a set of selected indices curinds.
def generateBatch(curinds,elements,atomArraysAll,nAtomsDict,atomsIndsReverse):
    atomArrays={}
    for element in elements:
        atomArrays[element]=[]
    atomArraysFinal={}
    
    curNAtoms={}
    for element in elements:
        validKeys=np.in1d(atomsIndsReverse[element],curinds)
        curNAtoms[element]=[]
        for ind in curinds:
            if len(atomArraysAll[element])>0:
                curNAtoms[element].append(len(atomArraysAll[element][ind]))
            else:
                curNAtoms[element].append(0)
        if len(validKeys)>0:
            atomArraysFinal[element]=atomArraysAll[element][validKeys]
        else:
            atomArraysFinal[element]=[]
    atomInds={}

    for element in elements:
        validKeys=np.in1d(atomsIndsReverse[element],curinds)
        if len(validKeys)>0:
            atomIndsTemp=np.sum(atomsIndsReverse[element][validKeys],1)
            atomInds[element]=atomIndsTemp*0.
            for i in range(len(curinds)):
                atomInds[element][atomIndsTemp==curinds[i]]=i
        else:
            atomInds[element]=[]
    return atomArraysFinal,atomInds

def generateTensorFlowArrays(images,fingerprintDB,elements,keylist):
    nAtomsDict={}
    for element in elements:
        nAtomsDict[element]=np.zeros(len(images))
        
    for j in range(len(keylist)):
        atoms=images[keylist[j]]
        fp=fingerprintDB[keylist[j]]
        for i in range(len(fp)):
            atom=atoms[i]
            nAtomsDict[atom.symbol][j]+=1
    
    atomsPositions={}
    for element in elements:
        atomsPositions[element]=np.cumsum(nAtomsDict[element])-nAtomsDict[element]

    atomsIndsReverse={}
    for element in elements:
        atomsIndsReverse[element]=[]
        for i in range(len(keylist)):
            if nAtomsDict[element][i]>0:
                atomsIndsReverse[element].append(np.ones((nAtomsDict[element][i],1))*i)
        if len(atomsIndsReverse[element])>0:
            atomsIndsReverse[element]=np.concatenate(atomsIndsReverse[element])
        
    atomArraysAll={}
    for element in elements:
        atomArraysAll[element]=[]

    energies=np.zeros((len(keylist),1))
    natoms=np.zeros((len(keylist),1))
    for j in range(len(keylist)):
        atoms=images[keylist[j]]
        fp=fingerprintDB[keylist[j]]
        for element in elements:
            atomArraysTemp=[]
            curatoms=[atom for atom in atoms if atom.symbol==element]
            for i in range(len(curatoms)):
                atomArraysTemp.append(fp[curatoms[i].index])
            if len(atomArraysTemp)>0:
                atomArraysAll[element].append(atomArraysTemp)
        energies[j]=atoms.get_potential_energy()
        natoms[j]=len(atoms)

    for element in elements:
        if len(atomArraysAll[element])>0:
            atomArraysAll[element]=np.concatenate(atomArraysAll[element])
        else:
            atomArraysAll[element]=[]
    return atomArraysAll,nAtomsDict,atomsIndsReverse,energies,natoms
