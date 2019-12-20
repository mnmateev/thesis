import sys
from mido import MidiFile, MidiTrack, Message
from keras.layers import LSTM, LeakyReLU, Dense, Activation, Dropout, Flatten
from keras.preprocessing import sequence
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from scipy.io.wavfile import read, write
import numpy as np
import pandas as pd
import pydub
from methods import *

#utility. calling this to create data from wavs

wav1 = sys.argv[1]
print(wav1+'.wav')
music1, music2, rate = parseWav(wav1, wav1)

#training
print('training data')
prepareData(pd.concat([music1.iloc[0:160000, :], music2.iloc[0:160000, :]], axis=0), wav1, wav1, look_back=3, train=True)
#evaluation data
print('testing data')
prepareData(pd.concat([music1.iloc[160001 : 400000, :], music2.iloc[160001 : 400000, :]], axis=0), wav1, wav1, look_back=3, train=False)
