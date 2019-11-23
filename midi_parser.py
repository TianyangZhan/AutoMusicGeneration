
'''
	This file contains utility functions to parse midi files
'''

import os
import numpy as np
from sklearn.utils import shuffle
from collections import defaultdict
from math import ceil
from mido import MidiFile, MidiTrack, Message, MetaMessage

# GLOBAL PARAMETERS
unit_time = 0.02 # unit: second 	# the time unit for each time slice (column in the piano roll)
highest_note = 81 # A_6		# pitch value
lowest_note = 33 # A_2		# pitch value
pitch_dimension = highest_note - lowest_note + 1


def parseMidi(midi_file):
	'''
		parse midi file into a piano roll and save temporal values

		params:		midi_file: a midi files for parsing
		
		output:	[pianoroll, tempo, resolution]
					pianoroll:	a matrix of size (timestep x pitch_dimension)
					tempo: the tempo value from midi file
					resolution: the resolution value from midi file

	'''

	midi_data = MidiFile(midi_file)

	# get music tempo info
	resolution = midi_data.ticks_per_beat

	track_tempos = [event.tempo for track in midi_data.tracks for event in track if str(event.type) == "set_tempo"]
	tempo = int(60000000/max(track_tempos)) # get the max track tempo

	ticks_per_time = resolution*tempo*unit_time/60.0 
	
	#Get maximum ticks across all tracks
	total_ticks =0
	for track in midi_data.tracks:
		sum_ticks = sum([event.time for event in track if str(event.type) in ['note_on','note_off','end_of_track']])		
		total_ticks = max(total_ticks,sum_ticks)

	time_slices = int(ceil(total_ticks/ticks_per_time))
	
	# slice file into piano roll matrix
	piano_roll = np.zeros((pitch_dimension, time_slices), dtype=int)
	note_states = defaultdict(lambda:-1)
	
	for track in midi_data.tracks:
		
		total_ticks = 0
		
		for event in track:

			if str(event.type) == 'note_on' and event.velocity > 0 and event.note in range(lowest_note,highest_note+1):
			# note is played

				total_ticks += event.time
				time_slice_idx = int(total_ticks/ticks_per_time)
				# count note as played
				note_idx = event.note - lowest_note
				piano_roll[note_idx][time_slice_idx] = 1
				note_states[note_idx] = time_slice_idx

			elif (str(event.type) == 'note_off' or str(event.type) == 'note_on') and event.note in range(lowest_note,highest_note+1): 
			# note is not played
				
				total_ticks += event.time
				time_slice_idx = int(total_ticks/ticks_per_time)

				if note_states[note_idx] != -1:	 # note was played
					piano_roll[note_idx][note_states[note_idx] : time_slice_idx] = 1
					note_states[note_idx] = -1

	return piano_roll.T, tempo, resolution


#preprocess data directory
def getData(data_path):
	'''
		parse midi files into a piano rolls

		params:		data_path: a folder that contains midi files
		
		output:		pianoroll_lst:	a list of (N = #files) matrices of size (timestep x pitch_dimension)

	'''
	print("Parsing MIDI files")

	pianoroll_lst = []
	
	for file in os.listdir(data_path):
		
		if not ( file.endswith(".midi") or file.endswith(".mid") ):
			continue

		pr,_,_ = parseMidi(os.path.join(data_path,file)) # don't need temporal values for training
		pianoroll_lst.append(pr)

	return pianoroll_lst


def createTrainData(pianoroll_lst, x_length, y_length, tight_window=False):
	'''
		create X and Y samples from piano roll matrix with a sliding window

		params:		pianoroll_lst: a list of piano roll matrices
					x_length: the length of input sequence. for best performance: x_length > y_length
					y_length: the length of output sequence. for best performance: y_length < x_length
					tight_window: default: False. the step size for shifting the sliding window.
								  tight_window=True: shift sliding window by y_length
								  tight_window=False: shift sliding window by x_length
		
		output:		[x,y]: shuffled data for training
	'''

	x = []
	y = []

	for piano_roll in pianoroll_lst:
		pos = 0
		while pos + x_length + y_length < piano_roll.shape[0]:
			x.append(piano_roll[pos:pos+x_length])
			y.append(piano_roll [pos+x_length: pos+x_length+y_length])
			if tight_window:
				pos += y_length
			else:
				pos += x_length

	return shuffle(np.array(x),np.array(y))


# NN output to pianoroll
def outputPianoRoll(output, note_threshold=0.1):
	'''
		convert a list of output to piano roll

		params:		output: a list of prediction result sequence
					note_threshold: default: 0.1. the threshold for a note to be played  
		
		output:		pianoroll_lst:	a list of matrices of size (timestep x pitch_dimension)
	'''
	pianoroll_lst = []
	for sequence in output:
		
		for timeslice in sequence:
			result = np.zeros(timeslice.shape)
			note_on = [i for i in range(len(timeslice)) if timeslice[i] > note_threshold]			
			result[note_on] = 1
			pianoroll_lst.append(result)

	return np.array(pianoroll_lst)
	

# pianoroll to MIDI
def outputMidi(output_dir, piano_roll, tempo = 120, resolution = 480, scale=1, velocity=65):
	'''
		convert the piano roll to midi file

		params:		output_dir: the directory to store output file
					piano_roll: a list of (N = #files) matrices of size (timestep x pitch_dimension)
					tempo: default: 120			the tempo value from midi file
					resolution: default: 480	the resolution value from midi file
					scale: default:1			the number of ticks per time slice.	for best performance: = length of sequence in one prediction
					velocity: default:65		the speed/strength to play a note
	'''

	
	ticks_per_time=(resolution*tempo*unit_time)/60.0

	mid = MidiFile(ticks_per_beat = int(resolution))
	
	track = MidiTrack()
	track.append(MetaMessage('set_tempo', tempo = int(60000000/tempo), time=0))

	note_events = ["note_off","note_on"]
	last_state = np.zeros(pitch_dimension)
	last_index = 0

	for current_index, current_state in enumerate(np.concatenate((piano_roll, last_state.reshape(1, -1)), axis=0)): # terminate note at the end
		
		delta = current_state - last_state
		last_state = current_state
		
		for i in range(len(delta)):
			if delta[i] == 1 or delta[i] == -1: # play/stop note
				event = Message(note_events[delta[i] > 0], time=int(scale*(current_index-last_index)*ticks_per_time), velocity=velocity, note=(lowest_note+i))
				track.append(event)
				last_index = current_index
			else:
				pass # don't change note state

	end = MetaMessage('end_of_track', time=1)
	track.append(end)
	
	mid.tracks.append(track)
	mid.save(output_dir)
