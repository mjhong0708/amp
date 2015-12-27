from tfAmpNN import tfAmpNN
import pickle
import tensorflow as tf
elementFPLengths={'Rh': 56, 'O': 56,'C':56,'H':56}


print('starting get_energy test!!')
keylist=images.keys()
for key  in keylist:
    energypredict=tfAmpNN_instance.get_energy(key,images,fingerprints)[0][0]
    expenergy=images[key].get_potential_energy()
    print('exp: %1.2f, pred: %1.2f, dE=%1.3f'%(expenergy,energypredict,energypredict-expenergy))