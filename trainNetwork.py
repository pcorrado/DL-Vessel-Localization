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
import ReinforcementCNN.Model as reinforcementModel
import OneShotCNN.DataGenerator as oneShotGenerator
import argparse

""" trainNetwork - trains the CNN for Deep Learning Based 4D Flow Plane Placement
        usage:
            python trainNetwork      (Trains ResNet)
"""
if __name__ == '__main__':
    saveName = "TrainedModel/ResNet_32_stride_16_batch_64"

    # Read in image paths and plane locations
    cutPlaneFileName = './data/trainCutPlaneList.csv'
    (images,labels) = readPlaneLocations(cutPlaneFileName)

    trainParams = {'shuffle': True, 'scaleFraction': 0.06, 'intensityMultiplier':  0.2, \
        'noiseFactor': 0.05, 'shiftPixels': 12.0, 'rotateDegrees': [5.0,5.0,35.0], \
        'dim': (32,32,32,5), 'stride': (16,16,16), 'batch_size': 64}

    # Datasets
    images = np.array(images)

    # Generators
    training_generator = oneShotGenerator.DataGenerator(images, labels,  **trainParams)

    # Initiate, compile, train, and save model
    myModel = oneShotModel.MyModel(input_shape=trainParams['dim'])
    myModel.compile(loss=oneShotModel.myLossFunction)
    myModel.summary()
    myModel.fit(training_generator, callbacks=[oneShotModel.MyCallBack()])
    myModel.save(saveName)
