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
from tensorflow.keras.layers import Lambda, Dense, Flatten
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
#np.set_printoptions(threshold=sys.maxsize)
os.environ['CUDA_VISIBLE_DEVICES']="1"
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--folder", type="string", default="/clusterfs/ml4hep/vmikuni/SWDAN/parsed", help="Folder containing input files")
parser.add_option("--file", type="string", default="PYTHIA", help="Name of input file")
(flags, args) = parser.parse_args()

alternative_mc = 'evaluate_multiplicity'

data = [
    'train',
    'test',
    'evaluate',
    alternative_mc,    
]

files = {}
datasets = {}
for datum in data:
    datasets[datum] = {
        "X": h5.File(os.path.join(flags.folder, "{}_{}.h5".format(datum,flags.file)),"r")['data'][:],
        "y":h5.File(os.path.join(flags.folder, "{}_{}.h5".format(datum,flags.file)),"r")['pid'][:],
        }
    print("Loaded dataset {} with {} events".format(datum,datasets[datum]['y'].shape))


def Classifier(NFEAT=6):
    LAYERSIZE = [32,64,32]
    inputs = Input((NFEAT, ))
    layer = Dense(LAYERSIZE[0], activation='relu')(inputs)
    #Encoder
    for il in range(1,len(LAYERSIZE)):
        layer = Dense(LAYERSIZE[il], activation='linear')(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
    layer = Dropout(0.25)(layer)
    outputs = Dense(1, activation='sigmoid')(layer)
    return inputs,outputs

EPOCHS=100
LR=1e-3

inputs,outputs = Classifier()
model_baseline = Model(inputs=inputs,outputs=outputs)
opt = tensorflow.keras.optimizers.Adam(learning_rate=LR)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

model_baseline.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

_ = model_baseline.fit(datasets['train']['X'], datasets['train']['y'], epochs=EPOCHS, 
                       callbacks=[callback],
                       batch_size=256,validation_data=(datasets['test']['X'], datasets['test']['y']))

pred_baseline = model_baseline.predict(datasets['evaluate']['X'],batch_size=1000)
fpr, tpr, _ = roc_curve(datasets['evaluate']['y'],pred_baseline, pos_label=1)    
print("Baseline target performance AUC: {}, acc {}".format(auc(fpr, tpr),accuracy_score(datasets['evaluate']['y'],pred_baseline>0.5)))

pred_variation = model_baseline.predict(datasets[alternative_mc]['X'],batch_size=1000)
fpr, tpr, _ = roc_curve(datasets[alternative_mc]['y'],pred_variation, pos_label=1)    
print("Varied target performance AUC: {}, acc {}".format(auc(fpr, tpr),accuracy_score(datasets[alternative_mc]['y'],pred_variation>0.5)))

