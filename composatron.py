import sys
import mido
import numpy as np
from random import *
from mido import MidiFile, MidiTrack, Message
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from methods import *

wav1 = sys.argv[1] 
wav2 = sys.argv[2]

# read files into pandas dataframes
music1, music2, rate = parseWav(wav1, wav2)

print('preparing training data')
# 1 and 2 for simultaneous training of two short snippets.
# temporary to reduce training time while building model
X1 = loadData('Samples/'+wav1 + 'X')
X2 = loadData('Samples/'+wav2 + 'X')
y1 = loadData('Samples/'+wav1 + 'Y')
y2 = loadData('Samples/'+wav2 + 'Y')

print('preparing test data')
# ~16k for training, ~24k for testing
test1 = loadData('Samples/'+wav2 + 'TX')
test2 = loadData('Samples/'+wav1 + 'TX')

# shape data, LSTM likes 3d data
X1 = X1.reshape((-1, 1, 3))
X2 = X2.reshape((-1, 1, 3))
test1 = test1.reshape((-1, 1, 3))
test2 = test2.reshape((-1, 1, 3))

print('creating model')
# initialize model
model1 = loadModel('model1')
model1 = compileModel(model1)
model2 = loadModel('model2')
model2 = compileModel(model2)

epochs = sys.argv[3]
batch_size = sys.argv[4]

trainModel(model1, X1, y1, int(epochs), int(batch_size))
trainModel(model2, X2, y2, int(epochs), int(batch_size))

saveModel(model1, 'model1')
saveModel(model2, 'model2')

# 'prediction', our composition!
prediction1 = model1.predict(test1)
prediction2 = model2.predict(test2)

print('saving output to wav')

# write prediction into wav
write('Output/p14.wav', rate, pd.concat([pd.DataFrame(prediction1.astype('int16')), pd.DataFrame(prediction2.astype('int16'))], axis=1).values)

# saving the original music in wav format
write('Output/o14.wav',rate, pd.concat([music1.iloc[160001 : 400000, :], music2.iloc[160001 : 400000, :]], axis=0).values)



