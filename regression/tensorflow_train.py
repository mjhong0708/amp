#!/bin/python

import pickle
import ase
import numpy as np
import tensorflow as tf

#Start with a dictionary of images and hashs, as from the middle of calc.train() in the amp code.  image_fp_dump.pkl contains about 1000 images of various small molecules sitting on various rhodium surfaces.  Please don't mis-use this data.
images,fingerprints=pickle.load(open('./image_fp_dump_20.pkl','r'))

#Hardcoded for four element types to show how this works, but should be pretty easy to modify this for other systems
elements=['Rh','O','C','H']
elementFPLengths={'Rh': 56, 'O': 56,'C':56,'H':56}
elementFPScales={'Rh': 1.,'O': 1.,'C':1.,'H':1.}
networkList=[5,5]
activationType=tf.nn.relu

nAtomsDict={}
for element in elements:
    nAtomsDict[element]=np.zeros(len(images))

keylist=images.keys()
for j in range(len(images)):
    atoms=images[keylist[j]]
    fp=fingerprints[keylist[j]]
    for i in range(len(fp)):
        atom=atoms[i]
        nAtomsDict[atom.symbol][j]+=1

atomsPositions={}
for element in elements:
    atomsPositions[element]=np.cumsum(nAtomsDict[element])-nAtomsDict[element]

atomsInds={}
for element in elements:
    atomsInds[element]=[]
    for i in range(len(keylist)):
        atomsInds[element].append(np.arange(nAtomsDict[element][i])+atomsPositions[element][i])

atomArraysAll={}
for element in elements:
    atomArraysAll[element]=[]

energies=np.zeros((len(images),1))
natoms=np.zeros((len(images),1))
for j in range(len(images)):
    atoms=images[keylist[j]]
    fp=fingerprints[keylist[j]]
    for element in elements:
        atomArraysTemp=[]
        curatoms=[atom for atom in atoms if atom.symbol==element]
        for i in range(len(curatoms)):
            atomArraysTemp.append(fp[curatoms[i].index])  
        atomArraysAll[element].append(atomArraysTemp)
    energies[j]=atoms.get_potential_energy()
    natoms[j]=len(atoms)

energies=energies
energies=energies-np.mean(energies)
energyScale=np.mean(np.abs(energies))

#Since we're not going to precondition the neural network with the simulatedannealing solver, instead we need to scale the inputs so that they're approximately all [-1,1].  We assign a scale for each element to make this happen.  There's probably a more clever way of doing this.
for element in elements:
    elementFPScales[element]=np.max(np.max(atomArraysAll[element]))

#This comes from the MNIST tutorial on the tensorflow website.  We want to initialize weights and bias's to be non-zero so that we break symmetry, and make sure all nodes are on at the start.  These two variables just handle this process.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)

#This is the heart of the NN model.  We define a generic two-layer neural network that will be used for each atom type.  It would be very easy to generalize this for any number of layers, and for different number of neurons in each layer.  This example has a 2x2 hidden-layer network
#here we define the inputs into the system.  We have one large tensor for each atom type.
tensordict={}
indsdict={}
for element in elements:
    tensordict[element]=tf.placeholder("float", shape=[None, elementFPLengths[element]])
    indsdict[element]=tf.placeholder("int64", shape=[None])

#The output to be compared against.  This will be the list of known energies
y_ = tf.placeholder("float", shape=[None, 1])

#Define a probability for dropout
keep_prob = tf.placeholder("float")
nAtoms_in=tf.placeholder("float",shape=[None,1])
#define the batchsize
batchsizeInput=tf.placeholder("int32")
learningrate=tf.placeholder("float")

#Construct the neural network for each atom type
outdict={}
for element in elements:
    outdict[element]=model(tensordict[element],indsdict[element],keep_prob,batchsizeInput,networkList,activationType)

#Set the input scaling based on the one for rhodium
curtensorscale=elementFPScales['Rh']
curenergyscale=100000.

#The total energy is the sum of the energies over each atom type
keylist=elements
ytot=outdict[keylist[0]]
for i in range(1,len(keylist)):
    ytot=ytot+outdict[keylist[i]]

#Define a loss function, this is the MAE of the predicted energy
loss=tf.sqrt(tf.reduce_mean(tf.square(tf.sub(ytot,y_))))
lossPerAtom=tf.sqrt(tf.reduce_mean(tf.square(tf.div(tf.sub(ytot,y_),nAtoms_in))))

#Define a stochastic optimizer.  The optimizer automatically works over the entire variables space, which encompasses all of the neural networks
train_step=tf.train.AdamOptimizer(learningrate).minimize(loss)

#Start the session
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


#Batch gradient descent.  Do an optimization over 1000 epochs of the input set.  For each epoch, the set of training images is broken up into random batches defined by batchsize above.  The key assumption is that the batch is representative of the entire sample.  This allows us to make optimization steps very quickly (based on info from a small number of images).  This technique is very common in literature (look for batch SGD).  The fact that it's stochastic is actually good, because it allows for some robust behavior against local minima

batchsize=20


def generateBatch(curinds,elements,atomArraysAll,nAtomsDict):
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

def trainmodel(nepoch,trainingrate,keepprob):
    icount=1
    indlist=np.arange(len(images))
    for j in range(nepoch):
        np.random.shuffle(indlist)
        
        for i in range(int(len(images)/batchsize)):
            #For each batch, construct a new set of inputs
            curinds=indlist[np.arange(batchsize)+i*batchsize]
            
            atomArraysFinal,atomInds=generateBatch(curinds,elements,atomArraysAll,nAtomsDict)
            
            feedinput={}
            for element in elements:
                feedinput[tensordict[element]]=atomArraysFinal[element]/elementFPScales[element]
                feedinput[indsdict[element]]=atomInds[element]
            feedinput[y_]=energies[curinds]/energyScale
            feedinput[batchsizeInput]=batchsize
            feedinput[learningrate]=trainingrate
            feedinput[keep_prob]=keepprob
            feedinput[nAtoms_in]=natoms[curinds]

            #run a training step with the new inputs
            sess.run(train_step, feed_dict=feedinput)

            #Print the loss function every 100 evals.  Would be better to handle this by evaluating the loss function on the entire dataset, but batchsize is currently hardcoded at the moment
            if icount%100==0:
                print(lossPerAtom.eval(feed_dict=feedinput)*energyScale)
            icount=icount+1

trainmodel(100000,1.e-4,0.5)


