
'''
	This file loads a trained Seq2Seq LSTM model and generate music
'''

import sys
import time
import random
import glob
import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam, RMSprop

from midi_parser import *

# GLOBAL PARAMETERS
x_length = 100 # sample sequence length.
y_length = 10 # output sequence legth. 		Needs to be consistent with the value in train.py
iteration = 50 # number of iteration to generate new sequence. 		Final result length: y_length * itertaion

saved_model = "./saved_params/LSTM_model.json"
saved_weights = "./saved_params/LSTM_weights.hdf5"
sample_folder = "./samples"
output_folder = "./output"


def generate(input_data, tempo, resolution):
	'''
		generate new music and save to a midi file

		params:
				input_data: seed music pianoroll for music generation
				tempo: tempo value parsed from the seed music
				resolution: resolution value parsed from the seed music
	'''

	output_path = os.path.join(output_folder, "generated_%s.midi"%(time.strftime("%Y%m%d_%H_%M")))

	# randomly select a sequence from the seed music
	start = np.random.randint(0, input_data.shape[0]-1-x_length-iteration)
	pattern = np.array(input_data[start:start+x_length])

	prediction_output = []

	# concatenate all generated sequence
	for i in range(iteration):
		prediction = model.predict(pattern.reshape(1,pattern.shape[0],-1).astype(float)).reshape(y_length,-1) # generate sequence
		prediction_output.append(prediction)
		pattern = np.append(pattern[y_length:,], prediction, axis=0) # shift sliding window on input data

	print("output shape: ", np.array(prediction_output).shape)

	# convert sequence back to piano roll
	pianoroll = outputPianoRoll(np.array(prediction_output), note_threshold=0.1)
	print("pianoroll shape: ", pianoroll.shape)
	
	# convert piano roll back to midi
	outputMidi(output_path, pianoroll, tempo, resolution, scale=int(y_length)) # scale: seqch output sequence has y_length ticks


if __name__ == '__main__':

	# load trained model
	model = model_from_json(open(saved_model).read())
	model.load_weights(saved_weights)
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer = RMSprop())

	# randomly select a file from sample folder
	midi_files = [file for file in os.listdir(sample_folder) if file.endswith(".midi") or file.endswith(".mid")]
	input_data, tempo, resolution = parseMidi( os.path.join(sample_folder, midi_files[random.randint(0,len(midi_files)-1)]) )

	# generate new music
	generate(input_data, tempo, resolution)
	