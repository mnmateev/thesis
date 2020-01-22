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

wav = sys.argv[1]

# read files into pandas dataframes
music, rate = parseWav(wav)

print('preparing training data')
# 1 and 2 for simultaneous training of two short snippets.
# temporary to reduce training time while building model
X = loadData('Samples/'+wav + 'X')
y = loadData('Samples/'+wav + 'Y')

print('preparing test data')
# ~16k for training, ~24k for testing
test = loadData('Samples/'+wav + 'TX')

# shape data, LSTM likes 3d data
X = X.reshape((-1, 1, 3))
test = test.reshape((-1, 1, 3))

print('creating model')
# initialize model
model = loadModel('model1')
model = compileModel(model)

epochs = sys.argv[2]
batch_size = sys.argv[3]

trainModel(model1, X, y, int(epochs), int(batch_size))

saveModel(model, 'model1')

# 'prediction', our composition!
prediction = model1.predict(test1)

print('saving output to wav')

# write prediction into wav
write('Output/p14.wav', rate, pd.DataFrame(prediction1.astype('int16')).values)

# saving the original music in wav format
write('Output/o14.wav',rate, music.iloc[160001 : 400000, :].values)



