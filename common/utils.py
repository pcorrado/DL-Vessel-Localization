""" Utility functions for data i/o and manipulation

Functions:
    readPlaneLocations - reads CSV file containing plane location and image paths
    readCase - reads in an image case
    augmentCase - perform data augmentation including random translation, rotation, scaling, intensity scaling, and noise addition
"""
import csv
import os
import math
import cv2
import numpy as np
from scipy.signal import resample
from scipy import ndimage
import matplotlib.pyplot as plt


""" readPlaneLocations - reads CSV file containing plane location and image paths
        inputs:
            filename - CSV file containing 8 columns:
                0: path to angiogram image
                1: vessel name
                2: plane x-position
                3: plane y-position
                4: plane z-position
                5: plane x-direction cosine * side length
                6: plane y-direction cosine * side length
                7: plane z-direction cosine * side length
        outputs:
            imageFileNames - Nx1 list of directory paths containing images (N=number of cases)
            labels  - Nx8x7 list plane positions, (8 vessels) with the following slice definitions:
                0: x-position
                1: y-position
                2: z-position
                3: side length
                4: x-direction cosine
                5: y-direction cosine
                6: z-direction cosine """
def readPlaneLocations(fileName):
    imageFileNames = []
    labels = {}

    with open(fileName, mode='r') as file:
        reader = csv.reader(file, delimiter=',')
        counter =0
        for row in reader:
            counter=counter+1
            if row and counter>1:
                path = row[0].replace('\\\\MPUFS7\\','/data/').replace('\\','/').replace('CD.dat','')
                vesselName = row[1]
                (cx, cy, cz) = (float(row[2]),float(row[3]),float(row[4]))
                (nx, ny, nz) = (float(row[5]),float(row[6]),float(row[7]))
                s = math.sqrt(nx**2+ny**2+nz**2)
                (nx,ny,nz) = (nx/s,ny/s,nz/s)
                if vesselName=='Aorta':
                    imageFileNames.append(path)
                    newLabels= [[cx, cy ,cz , s, nx, ny, nz]]
                else:
                    newLabels.append([cx, cy ,cz , s, nx, ny, nz])
                if vesselName=='LIPV':
                    labels[path]=np.array(newLabels)
        return (imageFileNames,labels)


""" readCase - reads in an image case
        inputs:
            baseDir - path to directory containing recontsructed chest PC VIPR image set
                must include the following files: MAG.dat, CD.dat, comp_vd_1.dat, comp_vd_2.dat, comp_vd_3.dat
        outputs:
            imageCase - targetSizex5 numpy array of image data. The 5 channels have the following images:
                0: time-averaged magnitude image
                1: time-averaged angiogram image
                2: time-averaged x-velocity image
                3: time-averaged y-velocity image
                4: time-averaged z-velocity image """
def readCase(baseDir, targetSize=(128,128,128)):
    headerFile = os.path.join(baseDir,'pcvipr_header.txt')
    for line in open(headerFile,'r'):
        (field,value) = line.split()
        if field=='matrixx': sizeX=int(float(value))
        if field=='matrixy': sizeY=int(float(value))
        if field=='matrixz': sizeZ=int(float(value))

    mag = loadDat(os.path.join(baseDir,'MAG.dat'),[sizeZ,sizeY,sizeX],targetSize=targetSize)
    cd = loadDat(os.path.join(baseDir,'CD.dat'),[sizeZ,sizeY,sizeX],targetSize=targetSize)
    vx = loadDat(os.path.join(baseDir,'comp_vd_1.dat'),[sizeZ,sizeY,sizeX],targetSize=targetSize)
    vy = loadDat(os.path.join(baseDir,'comp_vd_2.dat'),[sizeZ,sizeY,sizeX],targetSize=targetSize)
    vz = loadDat(os.path.join(baseDir,'comp_vd_3.dat'),[sizeZ,sizeY,sizeX],targetSize=targetSize)

    return np.concatenate((np.expand_dims(mag,axis=3),np.expand_dims(cd,axis=3),np.expand_dims(vx,axis=3),np.expand_dims(vy,axis=3),np.expand_dims(vz,axis=3)), axis=3)

# reads the image from a .dat file (16-bit integer binary file)
def loadDat(fileName, size, targetSize):
    img = np.fromfile(fileName, dtype=np.int16)
    img = img.reshape(size)
    img = np.transpose(img,axes=(2,1,0))
    (tx,ty,tz) = targetSize
    img = resample(img,tx,axis=0)
    img = resample(img,ty,axis=1)
    return resample(img,tz,axis=2)


""" augmentCase - perform data augmentation including random translation, rotation, scaling, intensity scaling, and noise addition
        inputs:
            image - Nx x Ny x Nz x Nc image (Nc = number of channels = 5)
            label - Nvessel x 7 list plane positions with the following slice definitions:
                0-2: x-, y-, and z- position
                3: side length
                4-6: x-, y-, and z- direction cosine
            shift_val (default=(0,0,0)) - number of pixels to shift image in x-, y-, and z- directions
            rotate_val (default=(0,0,0)) - number of degrees to rotate image on x-, y-, and z- axes
            scale_val (default=(1.0,1.0,1.0)) - factor by which to scale the size of the image in each direction (i.e. 2.0 would zoom in by a factor of 2)
            intensity_val (default=1.0) - constant to multiply the image by (i.e. 2.0 would make the image 2x brighter)
            noise_val (default=0) - standard deviation of noise to add to the image (as a fraction of the image background noise STDev)
            verbose (default=False) - true to include extra output to command line for debugging purposes
        outputs:
            image2 - Nx x Ny x Nz x Nc augmented image
            label2 - Nvessel x 7 augmented list plane positions """
def augmentCase(image,label,shift_val=(0,0,0),rotate_val=(0,0,0),scale_val=1.0,intensity_val=1.0, noise_val=0.0, verbose=False):
    image2 = np.copy(image).astype(np.float32)
    label2 = np.copy(label)

    (nx, ny, nz, nc) = image.shape

    Tr = makeTranslationMatrix(shift_val)
    Trc = makeTranslationMatrix([-nx/2.0,-ny/2.0,-nz/2.0]) # Translate to image center
    Trc2 = makeTranslationMatrix([nx/2.0,ny/2.0,nz/2.0]) # Translate back
    S = makeScalingMatrix(scale_val)
    R = makeRotationMatrix(rotate_val)

    T = np.matmul(np.matmul(np.matmul(np.matmul(Tr,Trc2),R),S),Trc) # Combine all affine transorm matrices into one matrix

    # Apply the affine transformation (rotation + scale + shift) to the image
    for c in range(image.shape[3]):
        image2[:, :, :, c] = ndimage.interpolation.affine_transform(image[:, :, :, c], np.linalg.inv(T), order=1)
        image2[:, :, :, c] /= np.amax(image2[:, :, :, c])

    image2 *= intensity_val     # Apply intensity variation
    image2 += generateNoise(image2, noise_val) # Add noise


    # Apply the affine transformation (rotation + scale + shift) to the plane center
    c = np.transpose(label[:, 0:3])    # Grab plane center
    c = np.concatenate((c,np.ones((1,c.shape[1]))),axis=0)
    c2 = np.matmul(T,c) # multiply by affine transform
    label2[:, 0:3] = np.transpose(c[0:3,:])

    # Scale plane side length by largest scale factor
    label2[:,3] = label2[:,3]*max(scale_val[0],scale_val[1],scale_val[2])

    n = np.transpose(label2[:, 4:7]) # Grab plane normals
    n = np.concatenate((n,np.ones((1,n.shape[1]))),axis=0)
    n2 = normalizeLength(np.matmul(np.matmul(R,S),n)) # Apple rotation and scaling to plane normals
    label2[:, 4:7] = np.expand_dims(np.transpose(n2[0:3,:]),axis=0)

    if verbose:
        print('T = {}'.format(T))
        print('Pre-transformation plane centers:')
        print(c)
        print('Post-transformation plane centers:')
        print(c2)
        print('Pre-transformation plane normals:')
        print(np.transpose(n[0:3,:]))
        print('Post-transformation plane normals:')
        print(np.transpose(n2[0:3,:]))
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image[:,:,64,1])
        plt.subplot(1,2,2)
        plt.imshow(image2[:,:,64,1])
        plt.show()

    return image2, label2

def makeTranslationMatrix(delta):
    Tr = np.identity(4,dtype=np.float32)
    Tr[:3,3] = np.transpose(np.array(delta))
    return Tr

# make a rotation matriz based on theta which gives x-, y-, and z- axis rotation angles in degrees
def makeRotationMatrix(theta):
    Rx = np.identity(4,dtype=np.float32)
    Ry = np.identity(4,dtype=np.float32)
    Rz = np.identity(4,dtype=np.float32)
    R = np.identity(4,dtype=np.float32)
    theta = np.array(theta)
    Rx[1,1] = math.cos(theta[0]*np.pi/180.0)
    Rx[1,2] = math.sin(theta[0]*np.pi/180.0)
    Rx[2,1] = -math.sin(theta[0]*np.pi/180.0)
    Rx[2,2] = math.cos(theta[0]*np.pi/180.0)
    Ry[0,0] = math.cos(theta[1]*np.pi/180.0)
    Ry[0,2] = -math.sin(theta[1]*np.pi/180.0)
    Ry[2,0] = math.sin(theta[1]*np.pi/180.0)
    Ry[2,2] = math.cos(theta[1]*np.pi/180.0)
    Rz[0,0] = math.cos(theta[2]*np.pi/180.0)
    Rz[0,1] = math.sin(theta[2]*np.pi/180.0)
    Rz[1,0] = -math.sin(theta[2]*np.pi/180.0)
    Rz[1,1] = math.cos(theta[2]*np.pi/180.0)
    return np.matmul(Rz,np.matmul(Ry,Rx))

# Scale image in each direction
def makeScalingMatrix(scale_val):
    S = np.identity(4,dtype=np.float32)
    scale_val = np.array(scale_val)
    S[0,0] = scale_val[0]
    S[1,1] = scale_val[1]
    S[2,2] = scale_val[2]
    return S

# Divide normal vector by its length
def normalizeLength(n):
    nL = np.sqrt(n[0,:]**2 + n[1,:]**2 + n[2,:]**2)
    n[0,:] = n[0,:]/nL
    n[1,:] = n[1,:]/nL
    n[2,:] = n[2,:]/nL
    return n

# Generate additive noise with a given fraction of the image background standard deviation
def generateNoise(img, noiseFraction, backgroundThreshPrctile=10):
        mask = img < np.percentile(img,backgroundThreshPrctile) # Mask of background (air) for noise STDev calculation
        _,stdev = cv2.meanStdDev(img*mask)
        return np.random.normal(scale=(stdev*noiseFraction), size=img.shape)
