# Music-Generation-with-LSTM


# Readme work in progress

## Overview

Music composition is a difficult task for many people since it requires a decent understanding of music theory as well as chord progression. In this project we try to build a neutral network to generate new music. We train a Sequence-to-Sequence Recurrent Neural Network with long short-term memory (LSTM) that can learn chord progressions from music in the training data and generate brand new music sequences using a short music sample as prior.

## Requirement:

**Python version:** python 3.x

**Required Packages:**

    pip install tensorflow
    pip install keras
    pip install mido
    pip install sklearn
    pip install numpy
    pip install ipykernel

**Training on GPU**

We use Keras (TensorfFlow backend), which automatically enables training on **GPU** if available*.

   1. Install GPU version of TensorFlow:
    
    pip install tensorflow-gpu
   
   2. Install CUDA:
    
    follow the instruction here: https://www.tensorflow.org/install/gpu
    

## Dataset:

We use MIDI as our input/output file type. 

The dataset (MAESTRO) used for the experiments is available at: https://magenta.tensorflow.org/datasets/maestro

## Train the Network: 

Training data: manually select midi files from the dataset and put in *./midi_songs* folder

    python train.py

## Generate Music: 

output file save to *./output* folder

    python generate.py

## Listen to The Music:

On macOS: Recommend using GaragBand to open and play midi files

There are also many online tools for playing midi files
