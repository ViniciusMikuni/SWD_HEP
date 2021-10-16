import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Lambda, Dense, Flatten, Conv2D,MaxPool2D
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.losses import binary_crossentropy, mse
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score
import sys
import pickle as pkl
from Disco import DisCo

import matplotlib.pyplot as plt


#np.set_printoptions(threshold=sys.maxsize)
os.environ['CUDA_VISIBLE_DEVICES']="3"
#os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
#tf.compat.v1.disable_eager_execution()
parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--folder", type="string", default="/clusterfs/ml4hep/vmikuni/SWDAN/parsed", help="Folder containing input files")
parser.add_option("--file", type="string", default="PYTHIA", help="Name of input file")
(flags, args) = parser.parse_args()


mnist = tf.keras.datasets.mnist
(x_train, y_mnist_train), (x_test, y_mnist_test) = mnist.load_data()

# Process MNIST
mnist_train = (x_train > 0).reshape(60000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (x_test > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

# Load MNIST-M
mnistm = pkl.load(open('../BSR/mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']

# mnistm_train = mnistm_train[y_mnist_train<2] #Make it binary for the moment
# mnistm_test = mnistm_test[y_mnist_test<2] #Make it binary for the moment

# mnist_train = mnist_train[y_mnist_train<2] #Make it binary for the moment
# mnist_test = mnist_test[y_mnist_test<2] #Make it binary for the moment

# y_mnist_train = y_mnist_train[y_mnist_train<2]
# y_mnist_test = y_mnist_test[y_mnist_test<2]

pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

mnist_train = (mnist_train-pixel_mean)/255.
mnistm_train = (mnistm_train-pixel_mean)/255.

mnist_test = (mnist_test-pixel_mean)/255.
mnistm_test = (mnistm_test-pixel_mean)/255.

def Classifier():
    LAYERSIZE = [64,128]
    inputs = Input((28,28,3))
    
    layer = Conv2D(LAYERSIZE[0], 5, activation='relu')(inputs)
    #Encoder
    for il in range(1,len(LAYERSIZE)):
        layer = Dense(LAYERSIZE[il], activation='relu')(layer)
        
    layer = MaxPool2D(2)(layer)
    layer = Flatten()(layer)
    
    layer = Dense(128, activation='relu')(layer)
    outputs = Dense(10, activation='softmax')(layer)

    return inputs,outputs


def Shuffle(X,y):
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)
    return X[idx],np.squeeze(y)[idx]


EPOCHS=4000
LR=1e-3


#Disco training

inputs,outputs = Classifier()
model_disco = Model(inputs=inputs,outputs=outputs)

opt = tensorflow.keras.optimizers.Adam(learning_rate=LR)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)


model_disco.compile(loss=DisCo, optimizer=opt, metrics=['accuracy'])

X_train, y_train = Shuffle(
    np.concatenate([mnist_train,mnistm_train],0),
    np.concatenate([y_mnist_train,10*np.ones(y_mnist_train.shape)],0),
)

X_test, y_test = Shuffle(
    np.concatenate([mnist_test,mnistm_test],0),
    np.concatenate([y_mnist_test,10*np.ones(y_mnist_test.shape)],0),
)



_ = model_disco.fit(
    X_train,y_train, 
    epochs=EPOCHS, 
    callbacks=[callback],
    batch_size=10000,validation_data=(X_test,y_test)
)


pred_baseline = model_disco.predict(mnist_test,batch_size=1000)
print("Disco domain acc {}".format(accuracy_score(y_mnist_test,np.argmax(pred_baseline,-1))))


pred_variation = model_disco.predict(mnistm_test,batch_size=1000)
print("Disco domain acc {}".format(accuracy_score(y_mnist_test,np.argmax(pred_variation,-1))))
