import numpy as np
import tensorflow as tf
import os
import cv2
import sys
from common.utils import augmentCase, readCase
from common.BaseDataGenerator import BaseDataGenerator
from math import floor



""" Reinforcement Network Data Generator for DL-based 4D flow cut-plane placement
        Generates patches of images of size (batch_size x Nx x Ny x Nz x Nc), where the five images
        channels are the magnitude, angiogram, and x-, y-, and z-velocity images.
        All images are ungated (time-averaged). Nx Ny Nz denote the patch size and are 32x32x32 by default.

        Also generates labels of size (batch_size x Nvessels x 8), where the 8 slices are scalars
        between 0 and 1 with the following meanings:
            0: distance from the true location (range 0-1)
            1: decision to move along x-direction. 0=move left; 0.5=stay; 1=move right.
            2: decision to move along y-direction. 0=move up; 0.5=stay; 1=move down.
            3: decision to move along z-direction. 0=move into plane; 0.5=stay; 1=move out of plane.
            4: plane side length normalized to the total image size (max of width, height, depth)
            5: x-direction cosine of plane normal rescaled from [-1,1] to [0,1]
            6: y-direction cosine of plane normal rescaled from [-1,1] to [0,1]
            7: z-direction cosine of plane normal rescaled from [-1,1] to [0,1]    """
class DataGenerator(BaseDataGenerator):

    """ DataGenerator initiator:
        Usage = DataGenerator(list_IDs, ...)
            Inputs:
                list_IDs - list of paths to image directories
                labels - dictionary with keys as image paths and values as Nvesselx7 arrays specifying plane locations
                batch_size    (default=128) - number of images per batch
                imageDims    (default=(128,128,128)) - target size to load images to
                dim    (default=(45,45,45,5)) - patch size (Nx x Ny x Nz x Nc)
                label_dim    (default=(8,8)) - output shape of labels (Nvessel x 8)
                shuffle    (default=True) - Shuffle order of images
                shiftPixels    (default=0) - stdev of number of pixels to shift image in each direction
                rotateDegrees   (default=(0,0,0)) - stdev of degrees by which to rotate image around x-, y-, and z- axis
                scaleFraction    (default=0.0) - fraction by which to scale image (1.0 doubles size of image) in each direction
                intensityMultiplier   (default=0.0) - fraction by which to increase or decrease image intensity
                noiseFactor   (default=0.0) - fraction of noise background to add to image
                stride    (default=(1,1,1)) - stride between successive patches
            Outputs:
                DataGenerator object"""
    def __init__(self, list_IDs, labels, imageDims=(128,128,128), batch_size=128, dim=(45,45,45,5), label_dim=(8,8), shuffle=True, \
                 shiftPixels=0, rotateDegrees=(0,0,0), scaleFraction=0.0, intensityMultiplier=0.0, noiseFactor=0.0, stride=(1,1,1), iterationsPerEpcoh=1000):
        # Initialization
        vars = locals()
        for name in vars:
            if not (name=='self'):
                setattr(self, name, vars[name])

        self.patches_per_image = ((self.imageDims[0]-self.dim[0])/self.stride[0]+1) * \
                                 ((self.imageDims[1]-self.dim[1])/self.stride[1]+1) * \
                                 ((self.imageDims[2]-self.dim[2])/self.stride[2]+1)

        self.indexes = np.arange(len(self.list_IDs))# Order of images

        # x, y, z, indexes into image to define patch locations from 0-(Npx-1), 0-(Npy-1), 0-(Npz-1) where Npx x Npy x Npz = patches_per_image
        self.xInd,self.yInd,self.zInd = np.meshgrid(range(int((self.imageDims[0]-self.dim[0])/self.stride[0]+1)),
                                                    range(int((self.imageDims[1]-self.dim[1])/self.stride[1]+1)),
                                                    range(int((self.imageDims[2]-self.dim[2])/self.stride[2]+1)))
        # Order of patches, length = patches_per_image
        self.subIndices = np.arange(self.patches_per_image, dtype=np.int32)
        # self.on_epoch_end()

    '__len__(dataGenerator) Denotes the number of batches per epoch'
    def __len__(self):
        return self.iterationsPerEpcoh
        #int(np.floor(len(self.list_IDs) * self.patches_per_image / self.batch_size))

    """ getImgNameXYZ
        Converts from within-patch x,y,z coordinates to within-image x,y,z coordinates.
        This is used during testing and validation to examine how accurately the network
        is working on an image-by-image basis rather than a patch-by-patch basis, as is done during training.
        Inputs:
            batch_index - number between 0 and (__len__(self)-1) indentifying which batch to look in
            patch_index - number between 0 and (batch_size-1) identifying which patch within the batch to look at
            x - x-position of plane center in patch coordinate system (0 to 1)
            y - y-position of plane center in patch coordinate system (0 to 1)
            z - z-position of plane center in patch coordinate system (0 to 1)
        Outputs:
            name - image directory path for the selected patch
            x2 - x-position of plane center in image coordinate system (0 to Nx-1)
            y2 - y-position of plane center in image coordinate system (0 to Ny-1)
            z2 - z-position of plane center in image coordinate system (0 to Nz-1)"""
    def getImgNameXYZ(self, batch_index, patch_index,x,y,z):
        index = self.indexes[int(np.floor((batch_index*self.batch_size+patch_index)/self.patches_per_image))]
        name = self.list_IDs[index]

        # Find index within patch based on patch_index and batch index
        leftover = batch_index*self.batch_size - np.floor(batch_index*self.batch_size/self.patches_per_image)*self.patches_per_image
        subIndex = self.subIndices[int((patch_index + leftover)%self.patches_per_image)]

        # Find x,y,z of patch upper left corner
        x1 = self.xInd.ravel()[subIndex]*self.stride[0]
        y1 = self.yInd.ravel()[subIndex]*self.stride[1]
        z1 = self.zInd.ravel()[subIndex]*self.stride[2]

        # Add the within-patch portion
        x2 = x*self.dim[0] + x1
        y2 = y*self.dim[1] + y1
        z2 = z*self.dim[2] + z1

        return name, x2, y2, z2

    """__getitem__ Generates one batch of data from batch_index.
        Inputs:
            batch_index - number of batch to generate
        Outputs:
            X - batch_size x Nx x Ny x Nz x Nc array of image patches
            y - batch_size x Nvesssel x 8 array of plane location information """
    def __getitem__(self, batch_index):
        # Generate indexes of the batch
        i1 = int(np.floor(batch_index*self.batch_size/self.patches_per_image))  # index of image 1
        i2 = int(np.floor((batch_index+1)*self.batch_size/self.patches_per_image)) # index of image Nimg+1

        np.random.shuffle(self.indexes)
        np.random.shuffle(self.subIndices)

        indexes = self.indexes[i1:(i2+1)] # indexes of image 1 through image Nimg - note, this does not have to have length of batch_size

        list_IDs_temp = [self.list_IDs[k] for k in indexes] # list of image paths
        subIndices = {} #
        counter=0
        for k in indexes:
            img_index = i1+counter
            si1 = max(0,batch_index*self.batch_size-img_index*self.patches_per_image)
            si2 = min(self.patches_per_image,(batch_index+1)*self.batch_size-img_index*self.patches_per_image)
            subIndices[self.list_IDs[k]] = self.subIndices[np.arange(si1,si2,dtype=np.int16)]
            counter+=1

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, subIndices)
        return X, y

    # 'Optionally shuffle batch and patch order after each epoch'
    # def on_epoch_end(self):
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)
    #         np.random.shuffle(self.subIndices)

    """__data_generation Generates one batch of data from list of image dirs and subIndices
        Inputs:
            batch_index - number of batch to generate
        Outputs:
            X - batch_size x Nx x Ny x Nz x Nc array of image patches
            y - batch_size x Nvesssel x 8 array of plane location information """
    def __data_generation(self, list_IDs_temp, subIndices):
        # Initialization
        XX = np.empty((self.batch_size, *self.dim))
        yy = np.empty((self.batch_size, *self.label_dim))

        counter = 0
        # Generate data
        for ID in list_IDs_temp:
            y = self.labels[ID]
            X = readCase(ID)
            X, y =  self.data_augmenter(X, y)
            for ind in subIndices[ID]:
                x0 = self.xInd.ravel()[ind]*self.stride[0] + (self.dim[0]/2.0)
                y0 = self.yInd.ravel()[ind]*self.stride[1] + (self.dim[1]/2.0)
                z0 = self.zInd.ravel()[ind]*self.stride[2] + (self.dim[2]/2.0)
                XX[counter,], yy[counter,] = self.getPatch(X, y, x0=x0, y0=y0, z0=z0)
                counter+=1
        return XX, yy

    def getPatch(self, img, label, x0=64,y0=64,z0=64):
        dx=(self.dim[0]/2.0)
        dy=(self.dim[1]/2.0)
        dz=(self.dim[2]/2.0)

        x0=np.amin([np.amax([x0,dx]),self.imageDims[0]-dx])
        y0=np.amin([np.amax([y0,dy]),self.imageDims[1]-dy])
        z0=np.amin([np.amax([z0,dz]),self.imageDims[2]-dz])


        XX = img[floor(x0-dx):floor(x0+dx),floor(y0-dy):floor(y0+dy),floor(z0-dz):floor(z0+dz),]
        yy = np.zeros((label.shape[0],8))
        yy[:,0] = np.sqrt( (label[:,0]-x0)**2 + (label[:,1]-y0)**2 + (label[:,2]-z0)**2 ) / max(self.imageDims[0:3])
        yy[:,1] = 0.5 + 0.5*(label[:,0] > x0) - 0.5*(label[:,0] < x0)
        yy[:,2] = 0.5 + 0.5*(label[:,1] > y0) - 0.5*(label[:,1] < y0)
        yy[:,3] = 0.5 + 0.5*(label[:,2] > z0) - 0.5*(label[:,2] < z0)
        yy[:,4] = label[:,3] / max(self.imageDims[0:3])
        yy[:,5:8] = (label[:,4:7] + 1.0) / 2.0
        return XX, yy

"For Testing purposes, remove once it is clearly working"
# if __name__ == '__main__':
#     cutPlaneFileName = './data/trainCutPlaneList.csv'
#     (images,labels) = readCSV(cutPlaneFileName)
#     images = images[0:2]
#     pdg = DataGenerator(images,labels,batch_size=1, dim=(64,64,64,2), stride=(32,32,32),label_dim=(8,8), shuffle= True, \
#               shiftPixels= 6.0, rotateDegrees= [5.0,5.0,12.0], scaleFraction= 0.05, intensityMultiplier= 0.2, noiseFactor= 0.05)
#     for batch in range(len(pdg)):
#         X,y = pdg.__getitem__(batch)
#         for patch_index in range(y.shape[0]):
#             if y[patch_index,0,0]>0.5:
#                 # print(patch_index)
#                 # print(pdg.subIndices)
#                 # print(pdg.subIndices[patch_index])
#                 # print(y[patch_index,0,])
#                 s = y[patch_index,0,4]*128
#                 (x0,y0,z0) = (y[patch_index,0,1]*64.0,y[patch_index,0,2]*64.0,y[patch_index,0,3]*64.0)
#                 (x1,x2,x3,y1,y2,y3) = getRowColumnVectors(y[patch_index,0,5:8]*2.0-1.0)
#                 (minX,maxX,minY,maxY,minZ,maxZ) = getPlaneBounds((x0,y0,z0),(x1,x2,x3),(y1,y2,y3),(64,64,64),s)
#                 (i, j) = np.meshgrid(np.linspace(start=-s/2.0, stop=s/2.0, num=32), np.linspace(start=-s/2.0, stop=s/2.0, num=32))
#
#                 xp = x0 + x1*i + y1*j
#                 yp = y0 + x2*i + y2*j
#                 zp = z0 + x3*i + y3*j
#                 # print(np.shape(X))
#                 # print(np.linspace(minX, maxX, maxX-minX+1))
#                 # print(np.linspace(minY, maxY, maxY-minY+1))
#                 # print(np.linspace(minZ, maxZ, maxZ-minZ+1))
#                 # print(np.shape(X[patch_index,minX:(maxX+1), minY:(maxY+1), minZ:(maxZ+1),1]))
#
#                 rgi = RegularGridInterpolator((np.linspace(minX, maxX, maxX-minX+1),
#                                                np.linspace(minY, maxY, maxY-minY+1),
#                                                np.linspace(minZ, maxZ, maxZ-minZ+1)),
#                                                np.squeeze(X[patch_index,minX:(maxX+1), minY:(maxY+1), minZ:(maxZ+1),1]),
#                                                bounds_error=False, fill_value=0)
#                 with warnings.catch_warnings():
#                     warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
#                     grid = np.transpose(np.reshape(rgi(np.transpose(np.array([xp.flatten(),yp.flatten(),zp.flatten()]))),(32,32)))
#                     plt.figure()
#                     plt.imshow(grid)
#                     plt.show()
#                 # name, x1, y1, z1 = pdg.getImgNameXYZ(batch, patch_index,y[patch_index,0,1],y[patch_index,0,2],y[patch_index,0,3])
#                 # print(name)
#                 # print('{}, {}, {}'.format(x1*128,y1*128,z1*128))
#                 # print('{}, {}, {}'.format(y[patch_index,0,5],y[patch_index,0,6],y[patch_index,0,7]))
