import numpy as np
import tensorflow as tf
import os
import cv2
import sys
from common.utils import augmentCase


class BaseDataGenerator(tf.keras.utils.Sequence):

    """ data_augmenter
        Perform online affine transformation on image and label using randomly sampled
        transformation parameters distributions with specified standard deviations.
            Inputs:
                image - Pre-augmentation 4D array of shape (Nx, Ny, Nz, Nc)
                label - Pre-augmentation 2D array of shape (Nvessel, 8)
            Outpus:
                image2 - Post-augmentation 4D array of shape (Nx, Ny, Nz, Nc)
                label2 - Post-augmentation 2D array of shape (Nvessel, 8)"""
    def data_augmenter(self, image, label):
        image2 = np.copy(image).astype(np.float32)
        label2 = np.copy(label).astype(np.float32)

        (nx, ny, nz, nc) = image.shape

        # Generate random affine transformation parameters using the Gaussian distribution
        shift_val = [np.clip(np.random.normal(), -3, 3) * self.shiftPixels/nx,
                     np.clip(np.random.normal(), -3, 3) * self.shiftPixels/ny,
                     np.clip(np.random.normal(), -3, 3) * self.shiftPixels/nz]

        rotate_val = [np.clip(np.random.normal(), -3, 3) * self.rotateDegrees[0],
                      np.clip(np.random.normal(), -3, 3) * self.rotateDegrees[1],
                      np.clip(np.random.normal(), -3, 3) * self.rotateDegrees[2]]

        scale_val = [1.0 + np.clip(np.random.normal(), -3, 3) * self.scaleFraction,
                     1.0 + np.clip(np.random.normal(), -3, 3) * self.scaleFraction,
                     1.0 + np.clip(np.random.normal(), -3, 3) * self.scaleFraction]
        intensity_val = 1.0 + np.clip(np.random.normal(), -3, 3) * self.intensityMultiplier
        noise_val = np.clip(np.random.normal(), 0.0, 3.0) * self.noiseFactor

        # Augment the image and label
        image2, label2 = augmentCase(image, label, shift_val, rotate_val, scale_val, intensity_val, noise_val)
        return image2, label2
