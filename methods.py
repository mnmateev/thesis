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

def mp3towav(filename):
	sound = pydub.AudioSegment.from_mp3(filename+".mp3")
	sound.export(filename+".wav", format="wav")

def parseWav(sample):
	# todo: parse whole song
	print('reading input')
	rate, music = read('Samples/'+sample+'.wav')

	# part of the song converted to a dataframe
	# default in scipy: 44,100 samples/sec
	print('parsing '+sample+'.wav')
	music = pd.DataFrame(music[0:800000, :])

	return music1, rate

# create training data by shifting the music data
# todo: save whole song
def prepareData(df, s, look_back=3, train=True):

	dataX, dataY = [],[]

	size = range(len(df)-look_back-1)
	for i in size:
		# 0 vs 1?
		dataX.append(df.iloc[i : i + look_back, 0].values)
		if train:
			dataY.append(df.iloc[i + look_back, 0])
	if train:
		np.save('Samples/'+s+'X.npy', dataX)
		np.save('Samples/'+s+'Y.npy', dataY)
		return np.array(dataX), np.array(dataY)
	else:
		np.save('Samples/'+s+'TX.npy', dataX)
		return np.array(dataX)

def initModel():
	print('creating model')
	model = Sequential()
	model.add(LSTM(units=128, activation='linear', input_shape=(None, 3)))
	model.add(LeakyReLU())
	model.add(Dense(units=64, activation='linear'))
	model.add(LeakyReLU())
	model.add(Dense(units=32, activation='linear'))
	model.add(LeakyReLU())
	model.add(Dense(units=16, activation='linear'))
	model.add(LeakyReLU())
	model.add(Dense(units=1, activation='linear'))
	model.add(LeakyReLU())

	return model

def compileModel(model):
	
	model.compile(optimizer='adam', loss='mean_squared_error')
	return model

def trainModel(model, X, y, epochs, batch_size):
	model.fit(X, y, epochs=epochs, batch_size = batch_size)

def saveModel(model, stri):
	# serialize model to JSON
	model_json = model.to_json()
	with open(stri+".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(stri+".h5")
	print("Saved model to disk")

def loadModel(model):
	# load json and create model
	json_file = open(model+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(model+".h5")
	print("Loaded model from disk")
	return loaded_model

def loadData(name):
	data = np.load(name + '.npy')
	print('loaded '+name+'.npy')
	return data



