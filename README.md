# DL-Vessel-Localization
Deep Learning for automatic vessel measurement plane placement on 4D flow MRI images.

This directory contains the source code used to train and test the network.

The network is based on Jihong Ju's implementation of 3D ResNet (See https://github.com/JihongJu/keras-resnet3d.)

A demo of this code can be found at https://share.streamlit.io/pcorrado/dl-vessel-localization
Please note that this demo shows vessel measurement planes placed in real time by the network, but does not show the GUI for manual plane placement used to annotate the training data. Rather, a very simplified interface is shown with three slider bars for navigating the image.
