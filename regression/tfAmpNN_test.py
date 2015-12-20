from tfAmpNN import tfAmpNN
import pickle
images,fingerprints=pickle.load(open('./image_fp_dump_20.pkl','r'))
elementFPLengths={'Rh': 56, 'O': 56,'C':56,'H':56}

tfAmpNN_instance=tfAmpNN(elementFingerprintLengths=elementFPLengths)
tfAmpNN_instance.train(images,fingerprints)