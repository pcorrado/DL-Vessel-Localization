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

""" testNetwork - test the CNN for Deep Learning Based 4D Flow Plane Placement
        usage:
            python testNetwork      (Tests ResNet)
"""
if __name__ == '__main__':

    modelName = "TrainedModel/TrainedModel_5Channels_Balanced/"
    outputFile = "TestResults/Balanced_Predictions.csv"

    # Load model
    myModel = oneShotModel.MyModel()
    myModel.load_model(modelName)

    # Read in test image paths and plane locations
    cutPlaneFileName = './data/O1CutPlaneList.csv'
    (images,labels) = readPlaneLocations(cutPlaneFileName)
    images = np.array(images)

    # Data generator
    test_generator = oneShotGenerator.DataGenerator(images, labels, shuffle=False, balanced=False)

    # Run predictions on test images
    predictedLocations, trueLocations = oneShotModel.MyModel.predictFullImages(myModel.model, test_generator, weightedAverage=False)
    dist, side, angle = oneShotModel.MyModel.comparePlaneLocations(predictedLocations, trueLocations)
    for vessel in range(dist.shape[0]):
        print('Vessel #{}: distance={}, side={}, angle={}\n'.format(vessel,np.mean(dist[vessel,:]),np.mean(side[vessel,:]),np.mean(angle[vessel,:])))

    # Write out plane locations to csv file
    with open(outputFile, mode='w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Path', 'Vessel', 'Center_X', 'Center_Y', 'Center_Z', 'Normal_X', 'Normal_Y', 'Normal_Z'])
        keys = list(predictedLocations.keys())
        for i,case in enumerate(keys):
            locPred = np.array(predictedLocations[case])
            length_pred = np.sqrt(np.sum((locPred[:,4:7]**2),axis=1))
            locPred[:,4] = locPred[:,4]/length_pred
            locPred[:,5] = locPred[:,5]/length_pred
            locPred[:,6] = locPred[:,6]/length_pred
            for (vessel, i) in [("Aorta",0),("MPA",1),("SVC",2),("IVC",3),("RSPV",4),("RIPV",5),("LSPV",6),("LIPV",7)]:
                writer.writerow([case, vessel, locPred[i,0], locPred[i,1], locPred[i,2], locPred[i,4]*locPred[i,3], locPred[i,5]*locPred[i,3], locPred[i,6]*locPred[i,3]])
