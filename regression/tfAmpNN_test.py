from tfAmpNN import tfAmpNN
import pickle
images,fingerprints=pickle.load(open('./image_fp_dump_20.pkl','r'))
elementFPLengths={'Rh': 56, 'O': 56,'C':56,'H':56}

tfAmpNN_instance=tfAmpNN(elementFingerprintLengths=elementFPLengths)
tfAmpNN_instance.train(images,fingerprints,nEpochs=10000)

print('starting get_energy test!!')
keylist=images.keys()
for key  in keylist:
    energypredict=tfAmpNN_instance.get_energy(key,images,fingerprints)[0][0]
    expenergy=images[key].get_potential_energy()
    print('exp: %1.2f, pred: %1.2f, dE=%1.3f'%(expenergy,energypredict,energypredict-expenergy))