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
def model(x,segmentinds,keep_prob,batchsize):
    nNeurons=2

    #Pass  the input tensors through the first soft-plus layer
    W_fc1 = weight_variable([elementFPLengths[element], nNeurons])
    b_fc1 = bias_variable([nNeurons])
    h_fc1 = tf.nn.softplus(tf.matmul(x, W_fc1) + b_fc1)

    #Define a drop-out layer (taken from the MNIST tutorial)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Pass the output of the first layer through the second layer
    nNeurons2=2
    W_fc2 = weight_variable([nNeurons, nNeurons2])
    b_fc2 = bias_variable([1])
    h_fc2=tf.nn.softplus(tf.matmul(h_fc1, W_fc2) + b_fc2)

    #The output will be a linear combination of the second layer outputs
    W_fc3 = weight_variable([ nNeurons2, 1])
    b_fc3 = bias_variable([1])
    y_out=tf.matmul(h_fc2, W_fc3) + b_fc3

    #Sum the predicted energy for each atom
    return tf.unsorted_segment_sum(y_out,segmentinds,batchsize)

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

#define the batchsize
batchsizeInput=tf.placeholder("int32")

#Construct the neural network for each atom type
outdict={}
for element in elements:
    outdict[element]=model(tensordict[element],indsdict[element],keep_prob,batchsizeInput)

#Set the input scaling based on the one for rhodium
curtensorscale=elementFPScales['Rh']
curenergyscale=100000.
#Test for just Rh

#The total energy is the sum of the energies over each atom type
keylist=elements
ytot=outdict[keylist[0]]
for i in range(1,len(keylist)):
    ytot=ytot+outdict[keylist[i]]

#Define a loss function, this is the MAE of the predicted energy
loss=tf.reduce_mean(tf.abs(tf.sub(ytot,y_)))

#Define a stochastic optimizer.  The optimizer automatically works over the entire variables space, which encompasses all of the neural networks
train_step=tf.train.AdamOptimizer(5e-5).minimize(loss)

#Start the session
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

def tfeval(obj):
    return obj.eval(feed_dict=feedinput)

#Batch gradient descent.  Do an optimization over 1000 epochs of the input set.  For each epoch, the set of training images is broken up into random batches defined by batchsize above.  The key assumption is that the batch is representative of the entire sample.  This allows us to make optimization steps very quickly (based on info from a small number of images).  This technique is very common in literature (look for batch SGD).  The fact that it's stochastic is actually good, because it allows for some robust behavior against local minima
icount=1
batchsize=20
indlist=np.arange(len(images))
for j in range(100000):
    np.random.shuffle(indlist)
    
    for i in range(int(len(images)/batchsize)):
        #For each batch, construct a new set of inputs
        curinds=indlist[np.arange(batchsize)+i*batchsize]
        
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

        atomsPositions={}
        for element in elements:
            atomsPositions[element]=np.cumsum(nAtomsDict[element][curinds])-nAtomsDict[element][curinds]
        
        atomInds={}
        for element in elements:
            atomInds[element]=np.zeros(np.sum(nAtomsDict[element][curinds]))
        
        for element in elements:
            curind=0
            for i in range(batchsize):
                for j in range(curNAtoms[element][i]):
                    atomInds[element][curind]=i
                    curind+=1

        feedinput={}
        for element in elements:
            feedinput[tensordict[element]]=atomArraysFinal[element]/curtensorscale
            feedinput[indsdict[element]]=atomInds[element]
        feedinput[y_]=energies[curinds]/curenergyscale
        feedinput[batchsizeInput]=batchsize

        #run a training step with the new inputs
        sess.run(train_step, feed_dict=feedinput)

        #Print the loss function every 100 evals.  Would be better to handle this by evaluating the loss function on the entire dataset, but batchsize is currently hardcoded at the moment
        if icount%100==0:
            print(loss.eval(feed_dict=feedinput)*curenergyscale)
        icount=icount+1



