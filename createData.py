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

#utility. calling this to create data from wavs or mp3

filename = sys.argv[1]

if(filename[:-3] == 'mp3'):
    # parse mp3 into wav
    mp3towav(filename[:-4])

wav = filename[:-4]

music, rate = parseWav(wav)

#training
print('training data')
prepareData(music.iloc[0:320000, :],wav1, look_back=3, train=True)
#evaluation data
print('testing data')
prepareData(music.iloc[320001 : 800000, :], wav1, look_back=3, train=False)
