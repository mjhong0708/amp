from tfAmpNN import tfAmpNN
import pickle
import tensorflow as tf
images,fingerprints=pickle.load(open('image_fp_dump_20.pkl','r'))

elementFPLengths={'Rh': 56, 'O': 56,'C':56,'H':56}

tfAmpNN_instance=tfAmpNN(elementFingerprintLengths=elementFPLengths,activation=tf.nn.relu,hiddenlayers=[5,5])
for i in range(5):
    trainingRate=1.e-3*10.**(-i)
    print(trainingRate)
    tfAmpNN_instance.train(images,fingerprints,batchsize=20,nEpochs=5000,initialTrainingRate=trainingRate,miniBatch=True)


print('starting get_energy test!!')
keylist=images.keys()
for key  in keylist:
    energypredict=tfAmpNN_instance.get_energy(key,images,fingerprints)[0][0]
    expenergy=images[key].get_potential_energy()
    print('exp: %1.2f, pred: %1.2f, dE=%1.3e'%(expenergy,energypredict,energypredict-expenergy))