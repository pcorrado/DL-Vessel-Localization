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
import YOLO.Model as yoloModel
import OneShotCNN.DataGenerator as oneShotGenerator
import YOLO.DataGenerator as yoloGenerator
import argparse

""" trainNetwork - trains one of the CNN's for Deep Learning Based 4D Flow Plane Placement
        usage:
            python trainNetwork      (Trains ResNet)
            python trainNetwork --yolo      (Trains YOLO Network)
"""
if __name__ == '__main__':
    # Parse command line input
    parser = argparse.ArgumentParser(description='Train DL Plane Placement Network.')

    parser.add_argument('--yolo', dest='model', action='store_const',
                       const=yoloModel, default=oneShotModel,
                       help='Use yolo network (default: Use resnet)')
    args = parser.parse_args()

    # Choose appropriate model and generator
    model = args.model
    generator = oneShotGenerator if model==oneShotModel else yoloGenerator
    saveName = "TrainedModel/OneShot_48_stride_20" if model==oneShotModel else "TrainedModel/YOLO_8grid"
    logName = "log/OneShot" if model==oneShotModel else "log/Yolo"

    # Read in image paths and plane locations
    cutPlaneFileName = './data/trainCutPlaneList.csv'
    (images,labels) = readPlaneLocations(cutPlaneFileName)

    # Parameters
    valParams = {'shuffle': False}
    if model==yoloModel:
        valParams['label_dim'] = (8,8,8,8,8)
        valParams['batch_size'] = 4
    else:
        valParams['dim'] =(48,48,48,5)
        valParams['stride'] = (20,20,20)
        valParams['batch_size'] = 16

    trainParams = {'shuffle': True, 'scaleFraction': 0.06, 'intensityMultiplier':  0.2, \
        'noiseFactor': 0.05, 'shiftPixels': 12.0, 'rotateDegrees': [5.0,5.0,35.0]}
    if model==yoloModel:
        trainParams['label_dim'] = (8,8,8,8,8)
        trainParams['batch_size'] = 4
    else:
        trainParams['dim'] =(48,48,48,5)
        trainParams['stride'] = (20,20,20)
        trainParams['batch_size'] = 16

    saveName = '{}_batch_{}'.format(saveName,trainParams['batch_size'])
    print(saveName)
    valCutoff=0.15

    order = np.random.permutation(len(images))
    cutoff = round((1-valCutoff)*len(images))

    # Datasets
    images = np.array(images)
    partition = {'train': images[order[0:cutoff]], 'validation': images[order[cutoff:]]}

    # Generators
    training_generator = generator.DataGenerator(partition['train'], labels,  **trainParams)
    validation_generator = generator.DataGenerator(partition['validation'], labels,  **valParams)
    # training_generator = validation_generator

    # Initiate, compile, train, and save model
    if model==yoloModel:
        myModel = model.MyModel(output_shape=trainParams['label_dim'])
    else:
        myModel = model.MyModel(input_shape=trainParams['dim'])
    myModel.compile(loss=model.myLossFunction)
    myModel.summary()
    myModel.fit(training_generator, validation_generator, callbacks=[model.MyCallBack(validation_data=validation_generator)])
    myModel.save(saveName)
