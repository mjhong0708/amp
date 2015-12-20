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
        tensordict={}
        indsdict={}
        for element in self.elements:
            tensordict[element]=tf.placeholder("float", shape=[None, self.elementFingerprintLengths[element]])
            indsdict[element]=tf.placeholder("int64", shape=[None])
        self.indsdict=indsdict
        self.tensordict=tensordict

        self.y_ = tf.placeholder("float", shape=[None, 1])
        self.keep_prob = tf.placeholder("float")
        self.nAtoms_in=tf.placeholder("float",shape=[None,1])
        self.batchsizeInput=tf.placeholder("int32")
        self.learningrate=tf.placeholder("float")

        outdict={}
        for element in self.elements:
            if isinstance(self.hiddenlayers,dict):
                networkListToUse=self.hiddenlayers[element]
            else:
                networkListToUse=self.hiddenlayers
            outdict[element]=model(tensordict[element],indsdict[element],self.keep_prob,self.batchsizeInput,networkListToUse,self.activation,self.elementFingerprintLengths[element])
        self.outdict=outdict

        #The total energy is the sum of the energies over each atom type
        keylist=self.elements
        ytot=outdict[keylist[0]]
        for i in range(1,len(keylist)):
            energy=ytot+outdict[keylist[i]]

        self.energy=energy
        self.loss=tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.energy,self.y_))))
        self.lossPerAtom=tf.sqrt(tf.reduce_mean(tf.square(tf.div(tf.sub(self.energy,self.y_),self.nAtoms_in))))

        self.train_step=tf.train.AdamOptimizer(self.learningrate).minimize(self.loss)

    def initializeVariables(self):
        self.sess.run(tf.initialize_all_variables())

    def train(self,images,fingerprintDB,batchsize=20):
        nAtomsDict={}
        for element in self.elements:
            nAtomsDict[element]=np.zeros(len(images))

        keylist=images.keys()
        for j in range(len(images)):
            atoms=images[keylist[j]]
            fp=fingerprintDB[keylist[j]]
            for i in range(len(fp)):
                atom=atoms[i]
                nAtomsDict[atom.symbol][j]+=1

        atomsPositions={}
        for element in self.elements:
            atomsPositions[element]=np.cumsum(nAtomsDict[element])-nAtomsDict[element]

        atomsInds={}
        for element in self.elements:
            atomsInds[element]=[]
            for i in range(len(keylist)):
                atomsInds[element].append(np.arange(nAtomsDict[element][i])+atomsPositions[element][i])

        atomArraysAll={}
        for element in self.elements:
            atomArraysAll[element]=[]

        energies=np.zeros((len(images),1))
        natoms=np.zeros((len(images),1))
        for j in range(len(images)):
            atoms=images[keylist[j]]
            fp=fingerprintDB[keylist[j]]
            for element in self.elements:
                atomArraysTemp=[]
                curatoms=[atom for atom in atoms if atom.symbol==element]
                for i in range(len(curatoms)):
                    atomArraysTemp.append(fp[curatoms[i].index])
                atomArraysAll[element].append(atomArraysTemp)
            energies[j]=atoms.get_potential_energy()
            natoms[j]=len(atoms)

        energies=energies
        energies=energies-np.mean(energies)
        self.energyScale=np.mean(np.abs(energies))

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

                    atomArraysFinal,atomInds=generateBatch(curinds,self.elements,atomArraysAll,nAtomsDict,batchsize)
                    feedinput={}
                    for element in self.elements:
                        feedinput[self.tensordict[element]]=atomArraysFinal[element]/self.elementFPScales[element]
                        feedinput[self.indsdict[element]]=atomInds[element]
                    feedinput[self.y_]=energies[curinds]/self.energyScale
                    feedinput[self.batchsizeInput]=batchsize
                    feedinput[self.learningrate]=trainingrate
                    feedinput[self.keep_prob]=keepprob
                    feedinput[self.nAtoms_in]=natoms[curinds]
                    
                    #run a training step with the new inputs
                    self.sess.run(self.train_step, feed_dict=feedinput)
                    
                    #Print the loss function every 100 evals.  Would be better to handle this by evaluating the loss function on the entire dataset, but batchsize is currently hardcoded at the moment
                    if icount%100==0:
                        print(self.lossPerAtom.eval(feed_dict=feedinput)*self.energyScale)
                icount=icount+1

        trainmodel(100000,1.e-4,0.5)

    #implement methods to get the energy and forces for a set of configurations
    #def get_energy():

    #def get_forces():





def model(x,segmentinds,keep_prob,batchsize,neuronList,activationType,fplength):
    
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
    return tf.unsorted_segment_sum(y_out,segmentinds,batchsize)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)

def generateBatch(curinds,elements,atomArraysAll,nAtomsDict,batchsize):
    atomArrays={}
    for element in elements:
        atomArrays[element]=[]
    atomArraysFinal={}
    
    curNAtoms={}
    for element in elements:
        curNAtoms[element]=[]
        for ind in curinds:
            if len(atomArraysAll[element][ind])>0:
                atomArrays[element].append(atomArraysAll[element][ind])
            curNAtoms[element].append(len(atomArraysAll[element][ind]))
        atomArraysFinal[element]=np.concatenate(atomArrays[element])

    atomInds={}
    for element in elements:
        atomInds[element]=np.zeros(np.sum(nAtomsDict[element][curinds]))

    for element in elements:
        curind=0
        for i in range(batchsize):
            for j in range(curNAtoms[element][i]):
                atomInds[element][curind]=i
                curind+=1

    return atomArraysFinal,atomInds
