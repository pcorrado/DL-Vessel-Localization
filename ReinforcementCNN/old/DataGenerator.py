import numpy as np
import tensorflow as tf
import os
import cv2
import sys
from common.utils import augmentCase, readCase
from common.BaseDataGenerator import BaseDataGenerator
import random
from scipy.interpolate import RegularGridInterpolator
import warnings

class DataGenerator(BaseDataGenerator):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=128, imageDims=(128,128,128,5), minSize=6, maxSize=26,
                 dim=(45,45,45,5), label_dim=(8,7), shuffle=True, shiftPixels=0, rotateDegrees=(0,0,0),
                 scaleFraction=0.0, intensityMultiplier=0.0, noiseFactor=0.0):
        # Initialization
        vars = locals()
        for name in vars:
            if not (name=='self'):
                setattr(self, name, vars[name])

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs))

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index = self.indexes[batch_index]

        # Generate data
        X, y = self.__data_generation(self.list_IDs[index])
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ID):
        'Generates data containing batch_size samples'

        # Initialization
        XX = np.empty((self.batch_size, *self.dim))
        yy = np.empty((self.batch_size, *self.label_dim))

        # Generate data
        y = self.labels[list_ID]
        X = readCase(list_ID)
        X, y =  self.data_augmenter(X, y)

        rgi = RegularGridInterpolator((np.linspace(0, self.imageDims[0]-1, self.imageDims[0]),
                                       np.linspace(0, self.imageDims[1]-1, self.imageDims[1]),
                                       np.linspace(0, self.imageDims[2]-1, self.imageDims[2]),
                                       np.linspace(0, self.imageDims[3]-1, self.imageDims[3])),
                                       X, bounds_error=False, fill_value=0)

        for ii in range(self.batch_size):
            xp,yp,zp,c,x0,y0,z0,s,nx,ny,nz = self.getRandomPatchMesh()

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
                XX[ii,] = np.reshape(rgi(np.transpose(np.array([xp.flatten(),yp.flatten(),zp.flatten(), c.flatten()]))),(self.dim[0],self.dim[1],self.dim[2],self.dim[3]))

            yy[ii,:,0] = 0.5 + 0.5*(y[:,0] > x0) - 0.5*(y[:,0] < x0)
            yy[ii,:,1] = 0.5 + 0.5*(y[:,1] > y0) - 0.5*(y[:,1] < y0)
            yy[ii,:,2] = 0.5 + 0.5*(y[:,2] > z0) - 0.5*(y[:,2] < z0)
            yy[ii,:,3] = 0.5 + 0.5*(y[:,3] > s) - 0.5*(y[:,3] > s)
            yy[ii,:,4] = 0.5 + 0.5*(y[:,4] > nx) - 0.5*(y[:,4] > nx)
            yy[ii,:,5] = 0.5 + 0.5*(y[:,5] > ny) - 0.5*(y[:,5] > ny)
            yy[ii,:,6] = 0.5 + 0.5*(y[:,6] > nz) - 0.5*(y[:,6] > nz)
        return XX, yy

    "Generate random patch location within image"
    def getRandomPatchInfo(self):
        x0 = np.random.randint(np.floor(self.dim[0]/2),self.imageDims[0]-np.floor(self.dim[0]/2)+1)
        y0 = np.random.randint(np.floor(self.dim[1]/2),self.imageDims[1]-np.floor(self.dim[1]/2)+1)
        z0 = np.random.randint(np.floor(self.dim[2]/2),self.imageDims[2]-np.floor(self.dim[2]/2)+1)

        s = np.random.randint(self.minSize,self.maxSize+1)

        a = np.random.uniform(-1.0,1.0)
        b = np.random.uniform(-1.0,1.0)
        while (a*a+b*b)>=1.0:
            a = np.random.uniform(-1.0,1.0)
            b = np.random.uniform(-1.0,1.0)
        nx = 2*a*np.sqrt(1-a*a-b*b)
        ny = 2*b*np.sqrt(1-a*a-b*b)
        nz = 1-2*(a*a+b*b)
        return x0,y0,z0,s,nx,ny,nz

    "Generate a random mesh grid with coordinates of a random patch within the image"
    def getRandomPatchMesh(self):
        x0,y0,z0,s,nx,ny,nz = self.getRandomPatchInfo()
        (x1,x2,x3,y1,y2,y3) = getRowColumnVectors(np.array((nx,ny,nz)))

        (i, j, k, c) = np.meshgrid(np.linspace(start=-s/2.0, stop=s/2.0, num=self.dim[0]),
                                   np.linspace(start=-s/2.0, stop=s/2.0, num=self.dim[1]),
                                   np.linspace(start=-s/2.0, stop=s/2.0, num=self.dim[2]),
                                   np.linspace(start=0, stop=self.imageDims[3]-1, num=self.imageDims[3]))

        xp = x0 + x1*i + y1*j + nx*k
        yp = y0 + x2*i + y2*j + ny*k
        zp = z0 + x3*i + y3*j + nz*k
        return xp,yp,zp,c,x0,y0,z0,s,nx,ny,nz

"Get row and column directions from the normal direction of a plane"
def getRowColumnVectors(n):
    while True:
        x = np.array([random.random(),random.random(),random.random()])
        x = x-np.dot(x,n)*n
        if np.dot(x,x)>0:
            break
    x = x/np.sqrt(np.dot(x,x))

    while True:
        y = np.array([random.random(),random.random(),random.random()])
        y = y-np.dot(y,n)*n
        y = y-np.dot(y,x)*x
        if np.dot(y,y)>0:
            break
    y = y/np.sqrt(np.dot(y,y))

    if np.dot(np.cross(x,y),n)<0:
        y=-y

    return (x[0],x[1],x[2],y[0],y[1],y[2])

if __name__ == '__main__':
    cutPlaneFileName = './data/trainCutPlaneList.csv'
    (images,labels) = readCSV(cutPlaneFileName)
    images = images[0:2]
    pdg = PatchDataGenerator(images,labels,batch_size=1, dim=(64,64,64,2), stride=(32,32,32),label_dim=(8,8), shuffle= True, \
              shiftPixels= 6.0, rotateDegrees= [5.0,5.0,12.0], scaleFraction= 0.05, intensityMultiplier= 0.2, noiseFactor= 0.05)
    for batch in range(len(pdg)):
        X,y = pdg.__getitem__(batch)
        for patch_index in range(y.shape[0]):
            if y[patch_index,0,0]>0.5:
                # print(patch_index)
                # print(pdg.subIndices)
                # print(pdg.subIndices[patch_index])
                # print(y[patch_index,0,])
                s = y[patch_index,0,4]*128
                (x0,y0,z0) = (y[patch_index,0,1]*64.0,y[patch_index,0,2]*64.0,y[patch_index,0,3]*64.0)
                (x1,x2,x3,y1,y2,y3) = getRowColumnVectors(y[patch_index,0,5:8]*2.0-1.0)
                (minX,maxX,minY,maxY,minZ,maxZ) = getPlaneBounds((x0,y0,z0),(x1,x2,x3),(y1,y2,y3),(64,64,64),s)
                (i, j) = np.meshgrid(np.linspace(start=-s/2.0, stop=s/2.0, num=32), np.linspace(start=-s/2.0, stop=s/2.0, num=32))

                xp = x0 + x1*i + y1*j
                yp = y0 + x2*i + y2*j
                zp = z0 + x3*i + y3*j
                # print(np.shape(X))
                # print(np.linspace(minX, maxX, maxX-minX+1))
                # print(np.linspace(minY, maxY, maxY-minY+1))
                # print(np.linspace(minZ, maxZ, maxZ-minZ+1))
                # print(np.shape(X[patch_index,minX:(maxX+1), minY:(maxY+1), minZ:(maxZ+1),1]))

                rgi = RegularGridInterpolator((np.linspace(minX, maxX, maxX-minX+1),
                                               np.linspace(minY, maxY, maxY-minY+1),
                                               np.linspace(minZ, maxZ, maxZ-minZ+1)),
                                               np.squeeze(X[patch_index,minX:(maxX+1), minY:(maxY+1), minZ:(maxZ+1),1]),
                                               bounds_error=False, fill_value=0)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
                    grid = np.transpose(np.reshape(rgi(np.transpose(np.array([xp.flatten(),yp.flatten(),zp.flatten()]))),(32,32)))
                    plt.figure()
                    plt.imshow(grid)
                    plt.show()
                # name, x1, y1, z1 = pdg.getImgNameXYZ(batch, patch_index,y[patch_index,0,1],y[patch_index,0,2],y[patch_index,0,3])
                # print(name)
                # print('{}, {}, {}'.format(x1*128,y1*128,z1*128))
                # print('{}, {}, {}'.format(y[patch_index,0,5],y[patch_index,0,6],y[patch_index,0,7]))
