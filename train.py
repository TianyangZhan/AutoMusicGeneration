
'''
	This file trains a Seq2Seq LSTM model to learn to play music
'''

import sys
import os
import time
import ipykernel
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

from midi_parser import getData, createTrainData

# GLOBAL PARAMETERS
highest_note = 81 # A_6 	Needs to be consistent with the value in midi_parser.py
lowest_note = 33 # A_2		Needs to be consistent with the value in midi_parser.py
pitch_dimension = highest_note - lowest_note + 1

# Model parameters
num_hidden = 512
x_length = 100
y_length = 10
batch_size = 64
num_epochs = 100

load_weights = True
data_path = "./midi_songs"
weight_path = "./saved_params/LSTM_weights.hdf5"
model_path = "./saved_params/LSTM_model.json"



def buildModel():
	'''Build a Seq2Seq LSTM model'''

	#encoder
	model = Sequential()
	model.add(LSTM(num_hidden, input_dim = pitch_dimension, return_sequences = True ))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(LSTM(num_hidden))
	model.add(RepeatVector(y_length))
	
	#decoder
	model.add(LSTM(num_hidden, return_sequences = True))
	model.add(Dropout(0.2))

	model.add(LSTM(num_hidden, return_sequences = True))
	model.add(Dropout(0.2))

	model.add(LSTM(num_hidden, return_sequences = True))
	model.add(Dropout(0.2))
	
	model.add(TimeDistributed(Dense(pitch_dimension, activation= 'softmax')))
	model.add(TimeDistributed(Dense(pitch_dimension, activation= 'softmax')))

	return model



if __name__ == '__main__':

	# prepare data for training
	pianoroll = getData(data_path)
	X,Y = createTrainData(pianoroll, x_length, y_length)

	# build model
	model = buildModel()
	model.summary()
	if load_weights:
		model.load_weights(weight_path)
	model.compile(loss='categorical_crossentropy', optimizer = RMSprop())

	# model callbacks
	checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=0, save_best_only=True, mode='auto') # save weights
	earlystop = EarlyStopping(monitor='loss', patience= 10, verbose=0, mode= 'auto') # terminate training
	history = History() # plot training loss

	# train the model
	hist = model.fit(X.astype(np.bool), Y.astype(np.bool), batch_size=batch_size, epochs=num_epochs, callbacks=[earlystop, history, checkpoint])

	# save trained model structure
	open(model_path, 'w').write(model.to_json())

	# plot training loss
	img = plt.figure(figsize=(6,5), dpi=75)
	plt.plot(hist.history['loss'])
	img.savefig("TrainingLoss.png", bbox_inches='tight')
