from __future__ import  absolute_import, division, print_function, unicode_literals
import six
from math import ceil, acos, pi
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ( Input, Activation, Dense, Flatten,
    Conv3D, AveragePooling3D, MaxPooling3D, Add, BatchNormalization, Dropout)
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
import numpy as np
from functools import reduce
import time

"""A class for plane localizion using a 3D ResNet.
Based on Jihong Ju's implementation (See https://github.com/JihongJu/keras-resnet3d.)   """
class MyModel():
    def __init__(self, log_dir="logs/log_oneShot", input_shape=(32,32,32,5), num_outputs=(8,8), reg_factor=1e-5):
        self.model = MyModel.build(input_shape, num_outputs, basic_block, [2,2,2,1,1], reg_factor=reg_factor)
        self.log_dir=log_dir

    def compile(self, optimizer=tf.keras.optimizers.Adam(1e-3), loss=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[ppv, npv, dist, side, angle])

    def fit(self, x, epochs=200, callbacks=None):
        self.model.fit(x=x, epochs=epochs, callbacks=callbacks) #tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)

    def summary(self,line_length=120):
        self.model.summary(line_length=line_length)

    def save(self,saveFile):
        self.model.save(saveFile)

    def load_model(self, saveFile):
        # self.model.load_weights(saveFile)
        self.model = tf.keras.models.load_model(saveFile, custom_objects={'ppv':ppv, 'npv': npv, 'dist': dist, 'side': side, 'angle': angle, 'myLossFunction': myLossFunction})


    @staticmethod
    def build(input_shape, outputShape, block_fn, repetitions, reg_factor):
        """Instantiate a vanilla ResNet3D keras model.

        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D array (volumetric images
            in batch) as input and returns a 3D array (prediction) as output.
        """
        nVes, nVal = outputShape
        num_outputs = nVes*nVal

        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        block_fn = _get_block(block_fn)
        input = Input(shape=input_shape)
        # # first conv
        # conv1 = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
        #                         strides=(2, 2, 2),
        #                         kernel_regularizer=l2(reg_factor)
        #                         )(input)
        # pool2 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
        #                      padding="same")(conv1)

        # repeat blocks
        block = input
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block3d(block_fn, filters=filters,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average pool and classification
        pool2 = AveragePooling3D(pool_size=(block.shape[1],
                                            block.shape[2],
                                            block.shape[3]),
                                 strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        dense = Dropout(0.5)(flatten1)
        numFCLayers=3
        for dNum in range(numFCLayers):
            dense = Dense(units=num_outputs*2**(numFCLayers-dNum-1),
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(reg_factor))(dense)
            if dNum==(numFCLayers-1):
                dense = Activation("sigmoid")(dense)
            else:
                dense = Activation("relu")(dense)
        outputs = tf.keras.layers.Reshape(outputShape)(dense)

        return Model(inputs=input, outputs=outputs)

    @staticmethod
    def predictFullImages(model, dataGenerator, weightedAverage=False):
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
        maxLikelihood = {}
        maxLocPredicted = {}
        maxLocTrue = {}
        t = time.time()
        oldName = ''
        for batch_index,(X,y) in enumerate(dataGenerator):
            # print('Validation batch # {} of {}'.format(batch_index,len(dataGenerator)))
            y2 = model.predict(X)
            for patch_index in range(y.shape[0]):
                for vessel in range(y.shape[1]):
                    name, xImg, yImg, zImg = dataGenerator.getImgNameXYZ(batch_index,patch_index,y2[patch_index,vessel,1],y2[patch_index,vessel,2],y2[patch_index,vessel,3])
                    if name != oldName:
                        print(name)
                        # print('Took {} seconds to process {}.'.format(time.time()-t,oldName))
                        oldName=name
                        t = time.time()
                    if not name in maxLikelihood.keys():
                        maxLikelihood[name] = [[0,0,0] for v in range(y.shape[1])]
                        maxLocPredicted[name] = [[[],[],[]] for v in range(y.shape[1])]
                    if (not maxLikelihood[name][vessel]) or (y2[patch_index,vessel,0] > maxLikelihood[name][vessel][2]):
                        index=2
                        if y2[patch_index,vessel,0] > maxLikelihood[name][vessel][1]:
                            if y2[patch_index,vessel,0] > maxLikelihood[name][vessel][0]:
                                index=0
                            else:
                                index=1
                        maxLikelihood[name][vessel].insert(index,y2[patch_index,vessel,0])
                        maxLocPredicted[name][vessel].insert(index,[xImg, yImg, zImg, y2[patch_index,vessel,4]*max(dataGenerator.imageDims), \
                                                y2[patch_index,vessel,5]*2.0-1.0, y2[patch_index,vessel,6]*2.0-1.0, y2[patch_index,vessel,7]*2.0-1.0])
                        maxLikelihood[name][vessel].pop(3)
                        maxLocPredicted[name][vessel].pop(3)
                        # if vessel==0:
                        #     print(np.array(maxLocPredicted[name][0][0],dtype=int))
                    name, xImg, yImg, zImg = dataGenerator.getImgNameXYZ(batch_index,patch_index,y[patch_index,vessel,1],y[patch_index,vessel,2],y[patch_index,vessel,3])
                    if not name in maxLocTrue.keys():
                        maxLocTrue[name] = [[0,0,0,0,0,0,0] for v in range(y.shape[1])]
                    if y[patch_index,vessel,0]:
                        maxLocTrue[name][vessel] = [xImg, yImg, zImg, y[patch_index,vessel,4]*max(dataGenerator.imageDims), \
                                                    y[patch_index,vessel,5]*2.0-1.0, y[patch_index,vessel,6]*2.0-1.0, y[patch_index,vessel,7]*2.0-1.0]

        for name in maxLikelihood.keys():
            for vessel in range(len(maxLikelihood[name])):
                if weightedAverage:
                    totalProb = maxLikelihood[name][vessel][0] + maxLikelihood[name][vessel][1] + maxLikelihood[name][vessel][2]
                    print(name)
                    # print(totalProb)
                    a = float(maxLikelihood[name][vessel][0])/totalProb
                    b = float(maxLikelihood[name][vessel][1])/totalProb
                    c = float(maxLikelihood[name][vessel][2])/totalProb
                    # print(a)
                    # print(b)
                    # print(c)
                    print(maxLocPredicted[name][vessel][0])
                    print(maxLocPredicted[name][vessel][1])
                    print(maxLocPredicted[name][vessel][2])
                    maxLocPredicted[name][vessel] = a*np.array(maxLocPredicted[name][vessel][0], dtype=float) + \
                                                    b*np.array(maxLocPredicted[name][vessel][1], dtype=float) + \
                                                    c*np.array(maxLocPredicted[name][vessel][2], dtype=float)
                    print(maxLocPredicted[name][vessel])
                else:
                    maxLocPredicted[name][vessel] = maxLocPredicted[name][vessel][0]
        return maxLocPredicted, maxLocTrue

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
            # print(locTrue.shape)
            dist[:,i] = np.sqrt(np.sum((locPred[:,0:3]-locTrue[:,0:3])**2,axis=1))
            side[:,i] =np.abs(locPred[:,3]-locTrue[:,3])
            length_pred = np.sqrt(np.sum((locPred[:,4:7]**2),axis=1))
            locPred[:,4] = locPred[:,4]/length_pred
            locPred[:,5] = locPred[:,5]/length_pred
            locPred[:,6] = locPred[:,6]/length_pred
            cosDPhi =  np.sum((locPred[:,4:7]*locTrue[:,4:7]),axis=1)
            angle[:,i] = np.arccos(np.minimum(np.maximum(cosDPhi,0.0),1.0))*180.0/np.pi
        return dist, side, angle

    @staticmethod
    def predictFullImage(model, image, input_shape, batch_size):
        """ Predicts plane location IN IMAGE COORDINATES for one image.
            Outputs:
                predictedLocations:
                    list with values being an Nvesselx7 array with columns defined as follows:
                        0-2: x-, y-, and z- locations of cut planes in image coordinates
                        3: plane side length in # of image voxels
                        4-6: x-, y-, and z- plane normals (range -1 to 1)"""

        maxLikelihood = np.zeros(8)
        maxLocPredicted = np.zeros((8,7))
        strideX = int(np.floor(input_shape[0]/2))
        strideY = int(np.floor(input_shape[1]/2))
        strideZ = int(np.floor(input_shape[2]/2))
        nx = np.floor((image.shape[0]-input_shape[0])/strideX) + 1
        ny = np.floor((image.shape[1]-input_shape[1])/strideY) + 1
        nz = np.floor((image.shape[2]-input_shape[2])/strideZ) + 1
        counter=0
        X = np.zeros((batch_size, input_shape[0],input_shape[1],input_shape[2],input_shape[3]))
        x1 = np.zeros(batch_size)
        x2 = np.zeros(batch_size)
        y1 = np.zeros(batch_size)
        y2 = np.zeros(batch_size)
        z1 = np.zeros(batch_size)
        z2 = np.zeros(batch_size)
        for ii in range(int(nx)):
            for jj in range(int(ny)):
                for kk in range(int(nz)):
                    x1[counter] = ii*strideX
                    x2[counter] = x1[counter] + input_shape[0]
                    y1[counter] = jj*strideY
                    y2[counter] = y1[counter] + input_shape[1]
                    z1[counter] = kk*strideZ
                    z2[counter] = z1[counter] + input_shape[2]
                    # print(int(x1[counter]),int(x2[counter]),int(y1[counter]),int(y2[counter]),int(z1[counter]),int(z2[counter]))
                    X[counter,] = np.expand_dims(image[int(x1[counter]):int(x2[counter]),int(y1[counter]):int(y2[counter]),int(z1[counter]):int(z2[counter]),:],axis=0)
                    counter+=1
                    if counter==batch_size:
                        pred = model.predict(X)
                        counter=0
                        for b in range(pred.shape[0]):
                            for vessel in range(pred.shape[1]):
                                if (pred[b,vessel,0] > maxLikelihood[vessel]):
                                    maxLikelihood[vessel] = pred[b,vessel,0]
                                    maxLocPredicted[vessel,0] = pred[b,vessel,1]*(x2[b]-x1[b]) + x1[b]
                                    maxLocPredicted[vessel,1] = pred[b,vessel,2]*(y2[b]-y1[b]) + y1[b]
                                    maxLocPredicted[vessel,2] = pred[b,vessel,3]*(z2[b]-z1[b]) + z1[b]
                                    maxLocPredicted[vessel,3] = pred[b,vessel,4]*max(image.shape[0:3])
                                    maxLocPredicted[vessel,4] = pred[b,vessel,5]*2.0-1.0
                                    maxLocPredicted[vessel,5] = pred[b,vessel,6]*2.0-1.0
                                    maxLocPredicted[vessel,6] = pred[b,vessel,7]*2.0-1.0

        return maxLocPredicted

class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=None):
        super().__init__()
        self.validation_data=validation_data

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr*0.94)
        print("\nEpoch %05d: Learning rate was %6.4f, now it's %6.4f." % (epoch, lr, lr*0.94))

        if self.validation_data is not None:
            maxLocPredicted, maxLocTrue = MyModel.predictFullImages(self.model, self.validation_data)
            dist, side, angle = MyModel.comparePlaneLocations(maxLocPredicted, maxLocTrue)
            for vessel in range(dist.shape[0]):
                print('Vessel #{}: distance={}, side={}, angle={}\n'.format(vessel,np.mean(dist[vessel,:]),np.mean(side[vessel,:]),np.mean(angle[vessel,:])))

def _bn_relu(input):
    """Helper to build a BN -> relu block."""
    norm = BatchNormalization(axis=4)(input)
    return Activation("relu")(norm)

def _conv_bn_relu3D(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f

def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        # activation = Dropout(0.2)(activation)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f

def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = ceil(input.shape[1] / residual.shape[1])
    stride_dim2 = ceil(input.shape[2] / residual.shape[2])
    stride_dim3 = ceil(input.shape[3] / residual.shape[3])
    equal_channels = residual.shape[4] == input.shape[4]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Conv3D(
            filters=residual.shape[4],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal", padding="valid",
            kernel_regularizer=l2(1e-4)
            )(input)
    return Add()([shortcut, residual])

def _residual_block3d(block_function, filters, kernel_regularizer, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input
    return f

def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4), is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer
                           )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv1)
        return _shortcut3d(input, residual)

    return f

def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

def myLossFunction(y_true, y_pred):
    loss =  y_true[:,:,0]*(y_true[:,:,1] - y_pred[:,:,1])**2 + \
            y_true[:,:,0]*(y_true[:,:,2] - y_pred[:,:,2])**2 + \
            y_true[:,:,0]*(y_true[:,:,3] - y_pred[:,:,3])**2 + \
       0.5*(y_true[:,:,0]*(y_true[:,:,4] - y_pred[:,:,4])**2) + \
       0.5*(y_true[:,:,0]*(y_true[:,:,5] - y_pred[:,:,5])**2) + \
       0.5*(y_true[:,:,0]*(y_true[:,:,6] - y_pred[:,:,6])**2) + \
       0.5*(y_true[:,:,0]*(y_true[:,:,7] - y_pred[:,:,7])**2) + \
       2.5*(y_true[:,:,0]+0.1)*(y_true[:,:,0] - y_pred[:,:,0])**2

    return tf.reduce_mean(loss)

def ppv(y_true, y_pred):
    wrong = y_true[:,:,0]*tf.abs(y_true[:,:,0]-y_pred[:,:,0])*tf.math.reciprocal_no_nan(tf.math.count_nonzero(y_true[:,:,0],dtype=tf.dtypes.float32))*tf.cast(tf.size(y_true[:,:,0]),tf.dtypes.float32)
    return tf.reduce_mean(100.0*(1-wrong))

def npv(y_true, y_pred):
    wrong = (1.0-y_true[:,:,0])*tf.abs(y_true[:,:,0]-y_pred[:,:,0])*tf.math.reciprocal_no_nan(tf.math.count_nonzero(1-y_true[:,:,0],dtype=tf.dtypes.float32))*tf.cast(tf.size(y_true[:,:,0]),tf.dtypes.float32)
    return tf.reduce_mean(100.0*(1-wrong))

def dist(y_true, y_pred):
    distance = y_true[:,:,0]*tf.sqrt(tf.square(y_true[:,:,1]-y_pred[:,:,1]) +
                       tf.square(y_true[:,:,2]-y_pred[:,:,2]) +
                       tf.square(y_true[:,:,3]-y_pred[:,:,3]))*32.0*tf.math.reciprocal_no_nan(tf.math.count_nonzero(y_true[:,:,0],dtype=tf.dtypes.float32))*tf.cast(tf.size(y_true[:,:,0]),tf.dtypes.float32)
    return tf.reduce_mean(distance)

def side(y_true, y_pred):
    side = y_true[:,:,0]*tf.abs(y_true[:,:,4]-y_pred[:,:,4])*128.0*tf.math.reciprocal_no_nan(tf.math.count_nonzero(y_true[:,:,0],dtype=tf.dtypes.float32))*tf.cast(tf.size(y_true[:,:,0]),tf.dtypes.float32)
    return tf.reduce_mean(side)

def angle(y_true, y_pred):
    length_pred = tf.math.sqrt((y_pred[:,:,5]*2.0-1.0)*(y_pred[:,:,5]*2.0-1.0) +
                               (y_pred[:,:,6]*2.0-1.0)*(y_pred[:,:,6]*2.0-1.0) +
                               (y_pred[:,:,7]*2.0-1.0)*(y_pred[:,:,7]*2.0-1.0))

    cosDPhi = ((y_true[:,:,5]*2.0-1.0)*(y_pred[:,:,5]*2.0-1.0)/length_pred +
               (y_true[:,:,6]*2.0-1.0)*(y_pred[:,:,6]*2.0-1.0)/length_pred +
               (y_true[:,:,7]*2.0-1.0)*(y_pred[:,:,7]*2.0-1.0)/length_pred)
    dPhi = y_true[:,:,0]*tf.math.acos(tf.math.minimum(tf.math.maximum(cosDPhi,-1.0),1.0))*180.0/np.pi*tf.math.reciprocal_no_nan(tf.math.count_nonzero(y_true[:,:,0],dtype=tf.dtypes.float32))*tf.cast(tf.size(y_true[:,:,0]),tf.dtypes.float32)
    return tf.reduce_mean(dPhi)
