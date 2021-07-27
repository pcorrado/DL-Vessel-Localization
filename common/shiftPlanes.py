import sys
import os
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import readPlaneLocations, readCase
from skimage.filters import threshold_mean
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
sys.path.append('M:\\pcorrado\\CODE')
sys.path.append('/export/home/pcorrado/CODE')
from ChestFlowSlicer.reslicePlane import loadCase4D, resliceVessel, getRowColumnVectors
import cv2

def shiftPlanes(name, label):
    (mag,cd,vX,vY,vZ,sizeX,sizeY,sizeZ,sizeT) = loadCase4D(name)
    newLabel = np.copy(label)
    for vNum, vessel in enumerate(['Aorta', 'MPA', 'SVC', 'IVC', 'RSPV', 'RIPV', 'LSPV', 'LIPV']):
        resliceVessel(name,mag,cd,vX,vY,vZ,sizeX,sizeY,sizeZ,sizeT, \
            label[vNum][0],label[vNum][1],label[vNum][2],label[vNum][3],label[vNum][4],label[vNum][5],label[vNum][6], vessel)
        cdPlane = np.load(os.path.join(name,vessel+'_CD.npy'))
        velPlane = np.load(os.path.join(name,vessel+'_velNormal.npy'))

        # segmentPlane(os.path.join(name,vessel+'_MAG.npy'))
        # flow, _, _ = computeFlow(os.path.join(name,vessel+'_SEG.npy'), getResolution(name)*label[vNum][3])
        # meanFlow.append(np.mean(flow*60.0/1000.0))
        os.remove(os.path.join(name,vessel+'_MAG.npy'))
        os.remove(os.path.join(name,vessel+'_CD.npy'))
        # os.remove(os.path.join(name,vessel+'_SEG.npy'))
        os.remove(os.path.join(name,vessel+'_velNormal.npy'))

        vBlur = cv2.GaussianBlur(np.mean(velPlane,axis=2),(9,9),0)
        maxInd = np.argmax(vBlur.flatten())
        (xM,yM) = np.unravel_index(maxInd, vBlur.shape)
        # vBlur[xM-2:xM+3,yM-2:yM+3] = np.min(vBlur)
        # cdPlane[xM-2:xM+3,yM-2:yM+3] = np.min(cdPlane)
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(cdPlane,cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(vBlur,cmap='gray')

        dx = -(xM-31.5)/64.0*label[vNum,3]
        dy = -(yM-31.5)/64.0*label[vNum,3]

        (x1,x2,x3,y1,y2,y3) = getRowColumnVectors(np.squeeze(label[vNum,4:7]))

        dX = -dx*x1 - dy*y1
        dY = -dx*x2 - dy*y2
        dZ = -dx*x3 - dy*y3

        newLabel[vNum, 0] = newLabel[vNum, 0] + dX
        newLabel[vNum, 1] = newLabel[vNum, 1] + dY
        newLabel[vNum, 2] = newLabel[vNum, 2] + dZ

        resliceVessel(name,mag,cd,vX,vY,vZ,sizeX,sizeY,sizeZ,sizeT, \
            newLabel[vNum][0],newLabel[vNum][1],newLabel[vNum][2],newLabel[vNum][3],newLabel[vNum][4],newLabel[vNum][5],newLabel[vNum][6], vessel)
        cdPlane2 = np.load(os.path.join(name,vessel+'_CD.npy'))
        velPlane2 = np.load(os.path.join(name,vessel+'_velNormal.npy'))
        vBlur2 = cv2.GaussianBlur(np.mean(velPlane2,axis=2),(9,9),0)
        os.remove(os.path.join(name,vessel+'_MAG.npy'))
        os.remove(os.path.join(name,vessel+'_CD.npy'))
        # os.remove(os.path.join(name,vessel+'_SEG.npy'))
        os.remove(os.path.join(name,vessel+'_velNormal.npy'))

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        # ax1.imshow(cdPlane,cmap='gray')
        # ax1.axis('off')
        # ax2.imshow(cdPlane2,cmap='gray')
        # ax2.axis('off')
        # pos = ax3.imshow(vBlur,cmap='jet')
        # ax3.axis('off')
        # fig.colorbar(pos, ax=ax3)
        # pos = ax4.imshow(vBlur2,cmap='jet')
        # ax4.axis('off')
        # fig.colorbar(pos, ax=ax4)
        # plt.show()

        # print('Side length = {0:.2f} voxels.'.format(label[vNum,3]))
        # print('Need to shift {0:.2f} pixels in x-dir (down) and {1:.2f} pixels in y-dir (right).'.format(dx, dy))

        # plt.show()

    return newLabel

    # for vNum, vessel in enumerate(['Aorta', 'MPA', 'SVC', 'IVC', 'RSPV', 'RIPV', 'LSPV', 'LIPV']):
    #     dist = np.sqrt((x[skel>0]-label[vNum,0])**2 + (y[skel>0]-label[vNum,1])**2 + (z[skel>0]-label[vNum,2])**2)
    #     minDistInd = np.argmin(dist)
    #     minDistInd = ind[minDistInd]
    #     newLabel[vNum, 0] = x.flatten()[minDistInd]
    #     newLabel[vNum, 1] = y.flatten()[minDistInd]
    #     newLabel[vNum, 2] = z.flatten()[minDistInd]
    #     newLabel[vNum, 3] = newLabel[vNum, 3]*1.1
    # return newLabel

if __name__ == '__main__':
    fileName = './TestResults/OneShot_Predictions_weighted_average.csv'
    with open('./TestResults/Weighted_Shifted_Predictions.csv', mode='w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        row = ['Path', 'Vessel', 'Center_X', 'Center_Y', 'Center_Z', 'Normal_X', 'Normal_Y', 'Normal_Z']
        print(row)
        writer.writerow(row)

        (images, labels) = readPlaneLocations(fileName)
        flow = np.zeros((len(images),8))
        for (iii, image) in enumerate(images):
            print('Case # {} of {}.'.format(iii+1, len(images)))
            labels[image] = shiftPlanes(image, labels[image])
            for vNum, vessel in enumerate(['Aorta', 'MPA', 'SVC', 'IVC', 'RSPV', 'RIPV', 'LSPV', 'LIPV']):
                row = [image, vessel, labels[image][vNum,0], labels[image][vNum,1], labels[image][vNum,2], \
                labels[image][vNum,3]*labels[image][vNum,4], labels[image][vNum,3]*labels[image][vNum,5], labels[image][vNum,3]*labels[image][vNum,6]]
                writer.writerow(row)
