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

def parseWav(sample1, sample2):
	# todo: parse whole song
	print('reading input')
	rate, music1 = read('Samples/'+sample1+'.wav')
	rate, music2 = read('Samples/'+sample2+'.wav')

	# part of the song converted to a dataframe
	# default in scipy: 44,100 samples/sec
	print('parsing '+sample1+'.wav')
	music1 = pd.DataFrame(music1[0:400000, :])
	music2 = pd.DataFrame(music2[400001:800000, :])

	return music1, music2, rate

# create training data by shifting the music data
# todo: save whole song
def prepareData(df, s1, s2, look_back=3, train=True):

	dataX1, dataX2 , dataY1 , dataY2 = [],[],[],[]

	size = range(len(df)-look_back-1)
	for i in size:

		dataX1.append(df.iloc[i : i + look_back, 0].values)
		dataX2.append(df.iloc[i : i + look_back, 1].values)
		if train:
			dataY1.append(df.iloc[i + look_back, 0])
			dataY2.append(df.iloc[i + look_back, 1])
	if train:
		np.save('Samples/'+s1+'X.npy', dataX1)
		np.save('Samples/'+s2+'X.npy', dataX2)
		np.save('Samples/'+s1+'Y.npy', dataY1)
		np.save('Samples/'+s2+'Y.npy', dataY2)
		return np.array(dataX1), np.array(dataX2), np.array(dataY1), np.array(dataY2)
	else:
		np.save('Samples/'+s1+'TX.npy', dataX1)
		np.save('Samples/'+s2+'TX.npy', dataX2)
		return np.array(dataX1), np.array(dataX2)

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



