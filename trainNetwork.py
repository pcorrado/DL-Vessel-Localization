import sys
import os
import csv
from math import sqrt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from common.utils import readPlaneLocations
import OneShotCNN.Model as oneShotModel
import OneShotCNN.DataGenerator as oneShotGenerator
import argparse

""" trainNetwork - trains the CNN for Deep Learning Based 4D Flow Plane Placement
        usage:
            python trainNetwork      (Trains ResNet)
"""
if __name__ == '__main__':
    saveName = "TrainedModel_4Channels_Unbalanced"

    # Read in image paths and plane locations
    cutPlaneFileName = './data/trainCutPlaneList.csv'
    (images,labels) = readPlaneLocations(cutPlaneFileName)

    trainParams = {'shuffle': True, 'scaleFraction': 0.06, 'intensityMultiplier':  0.2, \
        'noiseFactor': 0.05, 'shiftPixels': 12.0, 'rotateDegrees': [5.0,5.0,35.0], \
        'dim': (32,32,32,4), 'stride': (16,16,16), 'batch_size': 32, 'balanced': False}
    valParams=trainParams.copy()
    valParams['scaleFraction'] = 0
    valParams['intensityMultiplier'] = 0
    valParams['noiseFactor']=0
    valParams['shiftPixels']=0
    valParams['rotateDegrees']= [0,0,0]

    # Datasets
    images = np.array(images)
    choice = np.random.choice(range(images.shape[0]), size=(int(np.round(0.85*images.shape[0])),), replace=False)
    ind = np.zeros(images.shape[0], dtype=bool)
    ind[choice] = True
    rest = ~ind

    # Generators
    training_generator = oneShotGenerator.DataGenerator(images[ind], labels,  **trainParams)
    validation_generator = oneShotGenerator.DataGenerator(images[rest], labels,  **valParams)

    # Initiate, compile, train, and save model
    myModel = oneShotModel.MyModel(input_shape=trainParams['dim'])
    myModel.compile(loss=oneShotModel.myLossFunction)
    myModel.summary()
    myModel.fit(training_generator, callbacks=[oneShotModel.MyCallBack(validation_data=validation_generator)])
    myModel.save(saveName)
