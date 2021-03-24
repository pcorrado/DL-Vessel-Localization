import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2
import numpy as np
from common.utils import readCase
from ReinforcementCNN.DataGenerator import DataGenerator
from math import floor

# class MyModel(tf.keras.Model):
class MyModel():

    def __init__(self, log_dir="logs/log_oneShot", input_shape=(45,45,45,5), num_outputs=(8,8), reg_factor=1e-6):
        # super(MyModel, self).__init__()
        self.logDir = log_dir

        self.conv1 = tf.keras.layers.Conv3D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(reg_factor))
        self.conv2 = tf.keras.layers.Conv3D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(reg_factor))
        self.conv3 = tf.keras.layers.Conv3D(64, 4, activation='relu', padding='same', kernel_regularizer=l2(reg_factor))
        self.conv4 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(reg_factor))

        self.flatten = tf.keras.layers.Flatten()

        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))
        self.pool3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))

        self.dense1a, self.dense1b, self.dense1c, self.dense1d, self.reshape1 = MyModel.makeFCLayers(num_outputs[1])
        self.dense2a, self.dense2b, self.dense2c, self.dense2d, self.reshape2 = MyModel.makeFCLayers(num_outputs[1])
        self.dense3a, self.dense3b, self.dense3c, self.dense3d, self.reshape3 = MyModel.makeFCLayers(num_outputs[1])
        self.dense4a, self.dense4b, self.dense4c, self.dense4d, self.reshape4 = MyModel.makeFCLayers(num_outputs[1])
        self.dense5a, self.dense5b, self.dense5c, self.dense5d, self.reshape5 = MyModel.makeFCLayers(num_outputs[1])
        self.dense6a, self.dense6b, self.dense6c, self.dense6d, self.reshape6 = MyModel.makeFCLayers(num_outputs[1])
        self.dense7a, self.dense7b, self.dense7c, self.dense7d, self.reshape7 = MyModel.makeFCLayers(num_outputs[1])
        self.dense8a, self.dense8b, self.dense8c, self.dense8d, self.reshape8 = MyModel.makeFCLayers(num_outputs[1])

        self.concat = tf.keras.layers.Concatenate(axis=1)

    # def call(self, inputs, training=False):
        inputs = tf.keras.Input(shape=input_shape)

        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        x1 = self.dense1a(x)
        x1 = self.dense1b(x1)
        x1 = self.dense1c(x1)
        x1 = self.dense1d(x1)
        x1 = self.reshape1(x1)

        x2 = self.dense2a(x)
        x2 = self.dense2b(x2)
        x2 = self.dense2c(x2)
        x2 = self.dense2d(x2)
        x2 = self.reshape2(x2)

        x3 = self.dense3a(x)
        x3 = self.dense3b(x3)
        x3 = self.dense3c(x3)
        x3 = self.dense3d(x3)
        x3 = self.reshape3(x3)

        x4 = self.dense4a(x)
        x4 = self.dense4b(x4)
        x4 = self.dense4c(x4)
        x4 = self.dense4d(x4)
        x4 = self.reshape4(x4)

        x5 = self.dense5a(x)
        x5 = self.dense5b(x5)
        x5 = self.dense5c(x5)
        x5 = self.dense5d(x5)
        x5 = self.reshape5(x5)

        x6 = self.dense6a(x)
        x6 = self.dense6b(x6)
        x6 = self.dense6c(x6)
        x6 = self.dense6d(x6)
        x6 = self.reshape6(x6)

        x7 = self.dense7a(x)
        x7 = self.dense7b(x7)
        x7 = self.dense7c(x7)
        x7 = self.dense7d(x7)
        x7 = self.reshape7(x7)

        x8 = self.dense8a(x)
        x8 = self.dense8b(x8)
        x8 = self.dense8c(x8)
        x8 = self.dense8d(x8)
        x8 = self.reshape8(x8)

        # return self.concat([x1,x2,x3,x4,x5,x6,x7,x8])
        self.model = tf.keras.Model(inputs=inputs, outputs=self.concat([x1,x2,x3,x4,x5,x6,x7,x8]))

    def compile(self, optimizer=tf.keras.optimizers.Adam(1e-4), loss=None):
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.build(input_shape=(64,45,45,45,5))

    def fit(self, x, validation_data, callbacks=None):
        self.model.fit(x=x, validation_data=validation_data, epochs=40, callbacks=callbacks) #tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)

    def summary(self,line_length=120):
        self.model.summary(line_length=line_length)

    def save(self,saveFile):
        self.model.save(saveFile)

    @staticmethod
    def makeFCLayers(nCols):
        denseA = tf.keras.layers.Dense(512, activation='relu')
        denseB = tf.keras.layers.Dense(256, activation='relu')
        denseC = tf.keras.layers.Dense(128, activation='relu')
        denseD = tf.keras.layers.Dense(nCols, activation='sigmoid')
        reshape = tf.keras.layers.Reshape((1,nCols))

        return denseA, denseB, denseC, denseD, reshape

    @staticmethod
    def predictFullImages(model, dataGenerator):
        """ Predicts plane location IN IMAGE COORDINATES for all images in the data generator.
            This can be used during validation or during testing.
            Outputs:
                predictedLocations:
                    dictionary with keys being the directory path to the images,
                    and values being an Nvesselx7 array with columns defined as follows:
                        0-2: x-, y-, and z- locations of cut planes in image coordinates
                        3: plane side length in # of image voxels
                        4-6: x-, y-, and z- plane normals (range -1 to 1)
                trueLocations:
                    true locations of the cut planes in image coordinates, with same column definitions
        """

        uniqueCases = dataGenerator.list_IDs
        batch_index=1
        locPredicted = {}
        locTrue = {}
        for case in uniqueCases:
            # print('Validation case # {} of {}'.format(batch_index,len(uniqueCases)))
            batch_index += 1

            X = readCase(case)
            y = dataGenerator.labels[case]

            locPredicted[case] = [[] for v in range(y.shape[0])]
            locTrue[case] = [[] for v in range(y.shape[0])]

            "Loop until max iterations is reached or patch stops moving"
            maxIters=150
            minIters=10
            numGuesses=10
            minStepPixels=2

            for vessel in range(dataGenerator.label_dim[0]):
                x0 = [64]
                y0 = [64]
                z0 = [64]
                counter=0
                stepPixels=10
                while ((counter<maxIters) and (stepPixels>minStepPixels)) or (counter<minIters):
                    patch,yTrue = dataGenerator.getPatch(X,y,x0=x0[-1],y0=y0[-1],z0=z0[-1])
                    yPred = model.predict(np.expand_dims(patch,axis=0))
                    x0.append(x0[-1]+np.round(yPred[0,vessel,1]*3.0-0.5)-1)
                    y0.append(y0[-1]+np.round(yPred[0,vessel,2]*3.0-0.5)-1)
                    z0.append(z0[-1]+np.round(yPred[0,vessel,3]*3.0-0.5)-1)
                    s = yPred[0,vessel,4]*np.amax(dataGenerator.imageDims)
                    nx = yPred[0,vessel,5]*2.0-1.0
                    ny = yPred[0,vessel,6]*2.0-1.0
                    nz = yPred[0,vessel,7]*2.0-1.0

                    xOld =  x0[0] if len(x0)<numGuesses else x0.pop(0)
                    yOld =  y0[0] if len(y0)<numGuesses else y0.pop(0)
                    zOld =  z0[0] if len(z0)<numGuesses else z0.pop(0)
                    stepPixels = np.sqrt((xOld-x0[-1])**2 + (yOld-y0[-1])**2 + (zOld-z0[-1])**2)
                    # print('Case #{}, Vessel#{}, location = ({},{},{})'.format(batch_index-1, vessel, x0[-1], y0[-1], z0[-1]))
                    counter+=1
                locPredicted[case][vessel] = [x0[-1], y0[-1], z0[-1], s, nx, ny, nz]
                locTrue[case][vessel] = y[vessel,]
        return locPredicted, locTrue

    @staticmethod
    def comparePlaneLocations(predictedLocations, trueLocations):
            """ compares predicted vs. true locations in image coordinates
                Inputs:
                    predictedLocations:
                        dictionary with keys being the directory path to the images,
                        and values being an Nvesselx7 array with columns defined as follows:
                            0-2: x-, y-, and z- locations of cut planes in image coordinates
                            3: plane side length in # of image voxels
                            4-6: x-, y-, and z- plane normals (range -1 to 1)
                    trueLocations:
                        true locations of the cut planes in image coordinates, with same column definitions
                Outputs:
                    dist: NvesselxNcase array of distances in voxels between predicted and true planes
                    side: NvesselxNcase array of differences in side lengths between predicted and true planes
                    angle: NvesselxNcase array of angles in degrees between predicted and true plane normals"""
            keys = list(predictedLocations.keys())
            dist = np.zeros((len(predictedLocations[keys[0]]),len(keys)))
            side = np.zeros((len(predictedLocations[keys[0]]),len(keys)))
            angle = np.zeros((len(predictedLocations[keys[0]]),len(keys)))
            for i,case in enumerate(keys):
                locPred = np.array(predictedLocations[case])
                locTrue = np.array(trueLocations[case])
                dist[:,i] = np.sqrt(np.sum((locPred[:,0:3]-locTrue[:,0:3])**2,axis=1))
                side[:,i] =np.abs(locPred[:,3]-locTrue[:,3])
                length_pred = np.sqrt(np.sum((locPred[:,4:7]**2),axis=1))
                locPred[:,4] = locPred[:,4]/length_pred
                locPred[:,5] = locPred[:,5]/length_pred
                locPred[:,6] = locPred[:,6]/length_pred
                cosDPhi =  np.sum((locPred[:,4:7]*locTrue[:,4:7]),axis=1)
                angle[:,i] = np.arccos(np.minimum(np.maximum(cosDPhi,0.0),1.0))*180.0/np.pi
            return dist, side, angle

class MyCallBack(tf.keras.callbacks.Callback):
        def __init__(self, validation_data=None):
            super().__init__()
            self.validation_data=validation_data

        def on_epoch_end(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            tf.keras.backend.set_value(self.model.optimizer.lr, lr*0.94)
            print("\nEpoch %05d: Learning rate was %6.4f, now it's %6.4f." % (epoch, lr, lr*0.94))

            locPredicted, locTrue = MyModel.predictFullImages(self.model, self.validation_data)
            dist, side, angle = MyModel.comparePlaneLocations(locPredicted, locTrue)
            for vessel in range(dist.shape[0]):
                print('Vessel #{}: distance={}, side={}, angle={}\n'.format(vessel,np.mean(dist[vessel,:]),np.mean(side[vessel,:]),np.mean(angle[vessel,:])))

def myLossFunction(y_true, y_pred):
    loss =  (y_true[:,:,1] - y_pred[:,:,1])**2 + \
            (y_true[:,:,2] - y_pred[:,:,2])**2 + \
            (y_true[:,:,3] - y_pred[:,:,3])**2 + \
            10*tf.math.reciprocal_no_nan(y_true[:,:,0]) * \
            ((y_true[:,:,4] - y_pred[:,:,4])**2 + \
             (y_true[:,:,5] - y_pred[:,:,5])**2 + \
             (y_true[:,:,6] - y_pred[:,:,6])**2 + \
             (y_true[:,:,7] - y_pred[:,:,7])**2)

    return tf.reduce_mean(loss)


if __name__ == '__main__':
    myModel = MyModel()
    myModel.compile()
    myModel.summary()
