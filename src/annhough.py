"""
Created on Sun Oct 18 22:18:10 2015

@author: Tom Gresavage, t-gresavage@onu.edu
"""
# Standard Imports
import os
import sys
import csv
import logging
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import cPickle

# Image Processing Imports
import mahotas as mh
import numpy.linalg as la
from scipy.ndimage.filters import sobel, gaussian_filter, gaussian_filter1d
from scipy.misc import imresize
from scipy.ndimage import imread as sp_imread
from scipy.ndimage.interpolation import shift, zoom, rotate
from skimage import transform as tf
from leafhough import *

# Neural Network Imports
import theano
import theano.tensor as T
import lasagne.updates
import lasagne.objectives
import lasagne.nonlinearities as nonlinearities
from lasagne import layers as layers

# Misc Imports
import itertools as it
import warnings
from progressbar import ProgressBar, Bar, Percentage
from collections import defaultdict
from leafarea import *
from image_processing import *


class LeafNetwork(object):
    def __init__(self, data1D, data2D, targets, ref_image, train_prop=0.75, num_epochs=1000, stop_err=0.01,
                 d_stable=1e-6, n_stable=10, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, nonlinearity='tanh', n_1Dfilters=2, n_2Dfilters=2, n_conv=2, n_dense=2,
                 freeze_autoencoder=False, verbose=True, plotting=True, verbosity=3, shuffle=True, pretrain=True,
                 nets=None, name=None, save_plots=True, **kwargs):
        """
        Initializes and trains the networks with the given parameters
            :param data1D:     array-like
                            Array of curvature values on which to train and validate the 1D network whose dimensions
                            are (# samples, points/sample)
            :param data2D:     array-like
                            Array of images on which to train and validate the 2D network whose dimensions are
                            (# samples, channels/sample, height, width)
            :param targets:    array-like
                            Array of targets whose dimensions are (# samples, 2). The first and second element of each
                            row should specify the scale and angle targets respectively.
            :param train_prop: float, optional
                            A proportion specifying how to split the data between training and validation sets. By
                            default this number is 0.75 which splits the data into 75% training and 25% validation.
            :param num_epochs: int, optional
                            Number of epochs to train the data before exiting. Default value is 1000
            :param stop_err:   float, optional
                            A small number specifying the acceptable level of error. Once this error is achieved the
                            training loop will no longer train the network. Default value is 0.01.
            :param d_stable:   float, optional
                            A small number that, when used in conjunction with n_stable, determines when to stop
                            training the network to avoid overfitting. If the error remains below d_stable for more than
                             n_stable epochs training stops. Default value is 1e-06.
            :param learning_rate:  float, optional
                            A small number specifying the learning rate to use when training using Adam algorithm.
                            Default value is 0.001 following the suggestions of Adam developers.
            :param beta_1:         float, optional
                            A number close to 1 to be used when training with the Adam algorithm. Denotes the
                            exponential decay rate of the first moment estimator of the gradient. Default value is 0.9.
            :param beta_2:         float, optional
                            A number close to 1 to be used when training with the Adam algorithm. Denotes the
                            exponential decay rate of the second moment estimator of the gradient. Default is 0.999
            :param epsilon:        float, optional
                            A small positive number to be used with the Adam algorithm to avoid numerical error.
                            Default value is 1e-8.
            :param nonlinearity:   string, optional
                            A string specifying the type of nonlinearity each layer should have. Accepts the following
                            values: tanh, softplus, softmax, sigmoid, ReLU, linear, exponential, identity. By default
                            this is set to 'tanh'.
                            *Be careful when changing this parameter. Glorot suggests that activation functions should
                             be symmetric to avoid saturation and ensure effective backpropagation of error.
            :param n_1Dfilters:    int, sequence of ints, optional
                            Number of learnable filters each 1D convolutional layer should have. If an int, then all
                            layers have this number. If a sequence is given, then each value in the sequence specifies
                            the number of filters for each layer respectively. If the sequence contains fewer than
                            n_conv values then the values are cycled, with the last value always being used for the last
                            convolutional layer. are Default is 2.
            :param n_2Dfilters:    int, sequence of ints, optional
                            Number of learnable filters each 2D convolutional layer should have. If an int, then all
                            layers have this number. If a sequence is given, then each value in the sequence specifies
                            the number of filters for each layer respectively. If the sequence contains fewer than
                            n_conv values then the values are cycled, with the last value always being used for the last
                            convolutional layer. Default is 2.
            :param n_conv:         int, optional
                            Number of convolutional layers the networks should have. Must be at least 2 in order to
                            avoid excessive training times and accurate results. Default is 2
            :param n_dense:        int, optional
                            Number of dense, fully connected layers the networks should have. Must be at least 2 in
                            order to ensure accuracy. Default is 2.
            :param freeze_autoencoder: Bool, optional
                            Whether or not to freeze the autoencoding layers after pretraining. Default is True. Useless
                            unless pretrain is also set to true
            :param pretrain:Bool,
                            Whether or not to pretrain the convolutional layers. Use in conjunction with
                            freeze_autoencoder
            :param nets:    int, sequence of ints (optional)
                            What types of nets to build. If not specified, builds all of the following networks which
                            may later be called using their integer specification:
                                0: 1-D convolutional network using curvature data as input
                                1: 1-D convolutional network strictly without pretraining
                                2: Standard MLP network accepting curvature data as input. Has the same number of layers
                                and roughly the same hyperparameters as a 1-D convolutional network with one learnable
                                filter
                                3: 2-D convolutional network using image data as input and scale, angle output
                                4: 2-D convolutional network using image data as input and scale output
                                4: 2-D convolutional network using image data as input and angle output
            :param name:    string, optional
                            what name to give the network. If specified a name is automatically constructed using
                            the hyperparameters n_conv, n_1Dfilters, n_2Dfilters, n_dense.
            :param shuffle: Bool,
                            Whether to shuffle the training and validation data. Defaults to True.
            :param verbose: Bool, optional
                            Whether or not to print messages to the console. Default is set to True.
            :param plotting: Bool, optional
                            Whether or not to show plots of the training and validation errors for each network after
                            pretraining and training. Default is True.
            :param save_plots:  Bool
                            Whether or not to save the plots to a file. Plots saved this way are named after the network
                            the type of data plotted, and the time at which they were trained.
            :param verbosity: int, optional
                            Set the verbosity level.
                            0: low verbosity
                            3: maximum verbosity
            :return:        LeafNetwork,
                            A LeafNetwork object containing all data and methods needed to reconstruct and retrain a
                            network
        """
        self._nonlindict = {'tanh': nonlinearities.tanh, 'sigmoid': nonlinearities.sigmoid,
                            'softmax': nonlinearities.softmax, 'ReLU': nonlinearities.rectify,
                            'linear': nonlinearities.linear, 'exponential': nonlinearities.elu,
                            'softplus': nonlinearities.softplus, 'identity': nonlinearities.identity}
        assert n_conv > 1, "Too few convolutional layers. Must have at least 2.\nn_conv: %r" % (n_conv)
        assert n_dense >= 2, "Too few dense layers. Must have at least 2.\nn_dense: %r" % (n_dense)
        assert len(
            data1D.shape) == 2, "1D data shape error: training data has wrong number of dimensions.\nExpected 2 with format (# samples, points/sample) and instead got %r" % (
            data1D.shape)
        assert len(
            data2D.shape) == 4, "2D data shape error: training data has wrong number of dimensions.\nExpected 4 with format (# samples, channels/sample, height, width) and instead got %r" % (
            data2D.shape)

        self.ref_image = rgb_to_grey(ref_image)
        self._input1D_size = data1D.shape[-1]
        self._input2D_shape = data2D.shape[1:]

        for i in range(data1D.shape[0]):
            targets[i, 0] = np.log2(targets[i, 0])
            targets[i, 1] = targets[i, 1] / (2. * np.pi) - .5
        new_data = np.zeros((data1D.shape[0], 1, data1D.shape[1]))
        for i in range(data1D.shape[0]):
            new_data[i, 0, :] = data1D[i, :]

        # self._n_conv = n_conv
        self._n_conv = 4
        # self._n_conv = 2
        # self._n_dense = n_dense
        self._n_dense = 3
        # self._n_dense = 2
        pretrain = False
        try:
            iter(n_1Dfilters)
        except TypeError:
            n_1Dfilters = [n_1Dfilters]
        try:
            iter(n_2Dfilters)
        except TypeError:
            n_2Dfilters = [n_2Dfilters]

        # self._n_1Dfilters = list(n_1Dfilters)
        # self._n_1Dfilters = [8]
        self._n_1Dfilters = [2]
        # self._n_2Dfilters = list(n_2Dfilters)
        # self._n_2Dfilters = [8]
        self._n_2Dfilters = [1]

        self.netdir = os.path.dirname(__file__)
        try:
            self.logtime = time.time()
            self.logdir = os.path.join(self.netdir, '../logs/' + str(self.logtime))
            os.makedirs(self.logdir)
        except:
            raise

        try:
            self.statdir = os.path.join(self.netdir, '../stats/' + str(self.logtime))
            os.makedirs(self.statdir)
        except:
            raise

        try:
            self.datadir = os.path.join(self.netdir, '../data/')
            os.mkdir(self.datadir)
        except OSError:
            pass

        cPickle.dump(new_data, open(os.path.join(self.datadir, "data_1D.pkl"), "wb"))
        cPickle.dump(data2D, open(os.path.join(self.datadir, "data_2D.pkl"), "wb"))
        cPickle.dump(targets, open(os.path.join(self.datadir, "targets.pkl"), "wb"))
        X_train, X_train2D, y_train, X_val, X_val2D, y_val = self.split_data(data1D=new_data, data2D=data2D,
                                                                             targets=targets, train_prop=train_prop,
                                                                             shuffle=shuffle)

        if name is None:
            self.__name__ = str(self.logtime) + "__" + str(self._n_conv) + "CLayers_" + str(self._n_dense) + "DLayers"
        else:
            self.__name__ = name
        self.create_layers(n_1Dfilters=self._n_1Dfilters, n_2Dfilters=self._n_2Dfilters, n_conv=self._n_conv,
                           n_dense=self._n_dense, nonlinearity=nonlinearity,
                           freeze_autoencoder=freeze_autoencoder, verbose=verbose,
                           verbosity=verbosity, **kwargs)
        self.train_network(X_train=X_train, X_train2D=X_train2D, y_train=y_train, X_val=X_val, X_val2D=X_val2D,
                           y_val=y_val, num_epochs=num_epochs, stop_err=stop_err, d_stable=d_stable, n_stable=n_stable,
                           learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                           verbose=verbose, plotting=plotting, verbosity=verbosity, save_plots=save_plots,
                           pretrain=pretrain, **kwargs)

    def create_layers(self, n_1Dfilters=2, n_2Dfilters=6, n_conv=2, n_dense=3, nonlinearity='tanh',
                      freeze_autoencoder=False, nets=None, pretrain=True, **kwargs):
        """
        Builds 1- and 2-D Convolutional Networks using Theano and Lasagne
        """
        verbose = kwargs.get("verbose", False)
        verbosity = kwargs.get("verbosity", 2)
        logfile = os.path.join(self.logdir, "create_layers.log")
        logging.basicConfig(filename=logfile, level=logging.DEBUG)
        logging.debug('Creating layers...')
        self.layer_params = defaultdict(list)
        if nets is None:
            logging.debug('No networks specified: returning all networks.')
            self.__nets__ = [i for i in range(4)]
        else:
            try:
                self.__nets__ = list(nets)
            except TypeError:
                self.__nets__ = [nets]
            logging.debug('Returning networks: %s' % (self.__nets__))

        try:
            iter(n_1Dfilters)
        except TypeError:
            n_1Dfilters = [n_1Dfilters]
        try:
            iter(n_2Dfilters)
        except TypeError:
            n_2Dfilters = [n_2Dfilters]

        n_1Dfilters = list(n_1Dfilters)
        n_2Dfilters = list(n_2Dfilters)

        if 0 in self.__nets__ or 1 in self.__nets__:
            logging.debug('Creating 1D convolutional input variable')
            self.input_var = T.tensor3('inputs')
        if 3 in self.__nets__:
            logging.debug('Creating 2D convolutional input variable')
            self.input_var2D = T.tensor4('2D inputs')

        if pretrain:
            if 0 in self.__nets__ or 1 in self.__nets__:
                logging.debug('Creating theano 1D autoencoder variable')
                self.AE_target_var = T.tensor3('AE inputs')
            if 3 in self.__nets__:
                logging.debug('Creating theano 2D autoencoder variable')
                self.AE_target_var2D = T.tensor4('AE 2D targets')
        logging.debug('Creating theano target variable')
        self.target_var = T.matrix('targets')

        logging.debug('Setting nonlinearity: ' + nonlinearity)
        self._nonlinearity = self._nonlindict[nonlinearity]
        self.nonlinearity = nonlinearity
        logging.debug('Setting hyperparameters...')
        self._n_conv = n_conv
        self._n_dense = n_dense
        self._n_1Dfilters = n_1Dfilters
        self._n_2Dfilters = n_2Dfilters
        self._frozen = freeze_autoencoder

        pool_size = kwargs.get('pool_size', 2)
        dropout = kwargs.get('dropout', 0.5)
        filter_size = kwargs.get('filter_size', 3)
        if verbose:
            logging.debug('Layering networks...')
            print "Layering Networks..."

        if pretrain:
            if 0 in self.__nets__:
                self.AELayers = []
            if 3 in self.__nets__:
                self.AE2DLayers = []
        if 0 in self.__nets__:
            self.Conv1DLayers = []
        if 3 in self.__nets__:
            self.Conv2DLayers = []

        """
        Input Layers
        """
        logging.debug('Creating input layers')
        if 0 in self.__nets__:
            self.Conv1DLayers.append(layers.InputLayer((None, 1, self._input1D_size), input_var=self.input_var))
            if pretrain:
                self.AELayers.append(layers.InputLayer((None, 1, self._input1D_size), input_var=self.input_var))
        if 1 in self.__nets__:
            self.DConvLayers = layers.InputLayer((None, 1, self._input1D_size), input_var=self.input_var)
            self.DConvLayers = layers.batch_norm(layers.Conv1DLayer(self.DConvLayers, num_filters=1, filter_size=3,
                                                                    nonlinearity=self._nonlinearity))
        if 2 in self.__nets__:
            self.DenseLayers = layers.InputLayer((None, 1, self._input1D_size), input_var=self.input_var)
            logging.debug('1D input layers created. Creating 2D input layers...')
        if 3 in self.__nets__:
            self.Conv2DLayers.append(
                layers.InputLayer((None, self._input2D_shape[0], self._input2D_shape[1], self._input2D_shape[2]),
                                  input_var=self.input_var2D))
            if pretrain:
                self.AE2DLayers.append(
                    layers.InputLayer((None, self._input2D_shape[0], self._input2D_shape[1], self._input2D_shape[2]),
                                      input_var=self.input_var2D))

        """
        Batch Normalization
        """
        logging.debug('Creating batch normalization layers')
        if 0 in self.__nets__:
            if pretrain:
                self.AELayers.append(layers.BatchNormLayer(self.AELayers[-1]))
                self.Conv1DLayers.append(layers.BatchNormLayer(self.Conv1DLayers[-1], alpha=self.AELayers[-1].alpha,
                                                               beta=self.AELayers[-1].beta,
                                                               gamma=self.AELayers[-1].gamma,
                                                               mean=self.AELayers[-1].mean,
                                                               inv_std=self.AELayers[-1].inv_std))
            else:
                self.Conv1DLayers.append(layers.BatchNormLayer(self.Conv1DLayers[-1]))
        if 2 in self.__nets__:
            self.DenseLayers = layers.BatchNormLayer(self.DenseLayers)
        if 3 in self.__nets__:
            if pretrain:
                self.AE2DLayers.append(layers.BatchNormLayer(self.AE2DLayers[-1]))
                self.Conv2DLayers.append(layers.BatchNormLayer(self.Conv2DLayers[-1], alpha=self.AE2DLayers[-1].alpha,
                                                               beta=self.AE2DLayers[-1].beta,
                                                               gamma=self.AE2DLayers[-1].gamma,
                                                               mean=self.AE2DLayers[-1].mean,
                                                               inv_std=self.AE2DLayers[-1].inv_std))
            else:
                self.Conv2DLayers.append(layers.BatchNormLayer(self.Conv2DLayers[-1]))
        logging.debug('Batch norm layers successfully created.')
        for c in range(n_conv - 1):
            """
            Convolutional Layers
            """
            logging.debug('Creating convolutional layer %s of %s.' % (c, n_conv))
            if 0 in self.__nets__:
                if pretrain:
                    self.AELayers.append(
                        layers.Conv1DLayer(self.AELayers[-1], num_filters=n_1Dfilters[c % len(n_1Dfilters)],
                                           filter_size=filter_size,
                                           nonlinearity=self._nonlinearity))
                    self.Conv1DLayers.append(
                        layers.Conv1DLayer(self.Conv1DLayers[-1], num_filters=n_1Dfilters[c % len(n_1Dfilters)],
                                           filter_size=filter_size,
                                           W=self.AELayers[-1].W, b=self.AELayers[-1].b,
                                           nonlinearity=self._nonlinearity))
                else:
                    self.Conv1DLayers.append(
                        layers.Conv1DLayer(self.Conv1DLayers[-1], num_filters=n_1Dfilters[c % len(n_1Dfilters)],
                                           filter_size=filter_size,
                                           nonlinearity=self._nonlinearity))
                self.layer_params["Conv1D"].append(
                    {"type": "convolutional", "num_filters": self.Conv1DLayers[-1].num_filters,
                     "filter_size": self.Conv1DLayers[-1].num_filters,
                     "shape": layers.get_output_shape(self.Conv1DLayers[-1]), "stride": self.Conv1DLayers[-1].stride})
                if 2 in self.__nets__:
                    self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout),
                                                         num_units=layers.get_output_shape(self.Conv1DLayers[-1])[
                                                                       -1] * self._n_1Dfilters[c % len(n_1Dfilters)],
                                                         nonlinearity=self._nonlinearity)
                    self.layer_params["Dense"].append(
                        {"type": "dense", "num_units": self.DenseLayers.num_units,
                         "shape": layers.get_output_shape(self.DenseLayers)})
            if c is not 0 and 1 in self.__nets__:
                # DConvLayers already has a convolutional layer from the batch normalization step, so skip on the first iteration.
                self.DConvLayers = layers.Conv1DLayer(self.DConvLayers, num_filters=n_1Dfilters[c % len(n_1Dfilters)],
                                                      filter_size=filter_size, nonlinearity=self._nonlinearity)
            if 3 in self.__nets__:
                if pretrain:
                    self.AE2DLayers.append(
                        layers.Conv2DLayer(self.AE2DLayers[-1], num_filters=n_2Dfilters[c % len(n_2Dfilters)],
                                           filter_size=filter_size,
                                           nonlinearity=self._nonlinearity))
                    self.Conv2DLayers.append(
                        layers.Conv2DLayer(self.Conv2DLayers[-1], num_filters=n_2Dfilters[c % len(n_2Dfilters)],
                                           filter_size=filter_size,
                                           W=self.AE2DLayers[-1].W, b=self.AE2DLayers[-1].b,
                                           nonlinearity=self._nonlinearity))
                else:
                    self.Conv2DLayers.append(
                        layers.Conv2DLayer(self.Conv2DLayers[-1], num_filters=n_2Dfilters[c % len(n_2Dfilters)],
                                           filter_size=filter_size,
                                           nonlinearity=self._nonlinearity))
                self.layer_params["Conv2D"].append(
                    {"type": "convolutional", "num_filters": self.Conv2DLayers[-1].num_filters,
                     "filter_size": self.Conv2DLayers[-1].filter_size,
                     "shape": layers.get_output_shape(self.Conv2DLayers[-1]), "stride": self.Conv2DLayers[-1].stride})
            logging.debug('Output shapes after convolution %s of %s' % (c, n_conv))
            if 0 in self.__nets__:
                logging.debug("1D Conv: " + str(layers.get_output_shape(self.Conv1DLayers[-1])))
                if 2 in self.__nets__:
                    logging.debug("MLP: " + str(layers.get_output_shape(self.DenseLayers)))
            if 3 in self.__nets__:
                logging.debug("2D Conv: " + str(layers.get_output_shape(self.Conv2DLayers[-1])))
            if verbose:
                print "Output shapes after convolution %s of %s" % (c, n_conv)
                if 0 in self.__nets__:
                    print "1D Conv: ", layers.get_output_shape(self.Conv1DLayers[-1])
                    if 2 in self.__nets__:
                        print "MLP: ", layers.get_output_shape(self.DenseLayers)
                if 3 in self.__nets__:
                    print "2D Conv: ", layers.get_output_shape(self.Conv2DLayers[-1])
                print

            """
            Max Pooling Layers
            """
            logging.debug('Creating max pooling layer %s of %s.' % (c, n_conv))
            if pretrain:
                if 0 in self.__nets__:
                    self.AELayers.append(layers.MaxPool1DLayer(self.AELayers[-1], pool_size=pool_size))
                if 3 in self.__nets__:
                    self.AE2DLayers.append(layers.MaxPool2DLayer(self.AE2DLayers[-1], pool_size=pool_size))
            if 0 in self.__nets__:
                # self.Conv1DLayers.append(layers.MaxPool1DLayer(self.Conv1DLayers[-1], pool_size=pool_size))
                self.Conv1DLayers.append(
                    layers.Pool1DLayer(self.Conv1DLayers[-1], pool_size=pool_size, mode='average_exc_pad'))
                self.layer_params["Conv1D"].append(
                    {"type": "max pooling", "pool_size": self.Conv1DLayers[-1].pool_size,
                     "stride": self.Conv1DLayers[-1].stride,
                     "shape": layers.get_output_shape(self.Conv1DLayers[-1])})
                if 2 in self.__nets__:
                    self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout),
                                                         num_units=layers.get_output_shape(self.Conv1DLayers[-1])[-1] *
                                                                   n_1Dfilters[c % len(n_1Dfilters)],
                                                         nonlinearity=self._nonlinearity)
                    self.layer_params["Dense"].append(
                        {"type": "dense", "num_units": self.DenseLayers.num_units,
                         "shape": layers.get_output_shape(self.DenseLayers)})
            if 1 in self.__nets__:
                self.DConvLayers = layers.MaxPool1DLayer(self.DConvLayers, pool_size=pool_size)
            if 3 in self.__nets__:
                self.Conv2DLayers.append(layers.MaxPool2DLayer(self.Conv2DLayers[-1], pool_size=pool_size))
                self.layer_params["Conv2D"].append(
                    {"type": "max pooling", "pool_size": self.Conv2DLayers[-1].pool_size,
                     "stride": self.Conv2DLayers[-1].stride,
                     "shape": layers.get_output_shape(self.Conv2DLayers[-1])})

            logging.debug('Output shapes after pooling layer %s of %s' % (c, n_conv))
            if 0 in self.__nets__:
                logging.debug("1D Conv: " + str(layers.get_output_shape(self.Conv1DLayers[-1])))
                if 2 in self.__nets__:
                    logging.debug("MLP: " + str(layers.get_output_shape(self.DenseLayers)))
            if 3 in self.__nets__:
                logging.debug("2D Conv: " + str(layers.get_output_shape(self.Conv2DLayers[-1])))
            if verbose:
                print "Output shapes after pooling layer %s of %s" % (c, n_conv)
                if 0 in self.__nets__:
                    print "1D Conv: ", layers.get_output_shape(self.Conv1DLayers[-1])
                    if 2 in self.__nets__:
                        print "MLP: ", layers.get_output_shape(self.DenseLayers)
                if 3 in self.__nets__:
                    print "2D Conv: ", layers.get_output_shape(self.Conv2DLayers[-1])
                print
            if pretrain and freeze_autoencoder:
                logging.debug('Freezing weights and biases')
                if 0 in self.__nets__:
                    self.Conv1DLayers[-1].params[self.Conv1DLayers[-1].W].remove("trainable")
                    self.Conv1DLayers[-1].params[self.Conv1DLayers[-1].b].remove("trainable")
                if 3 in self.__nets__:
                    self.Conv2DLayers[-1].params[self.Conv2DLayers[-1].W].remove("trainable")
                    self.Conv2DLayers[-1].params[self.Conv2DLayers[-1].b].remove("trainable")
                    ###########################################################################################################
        """
        Last Convolutional Layers
        """
        logging.debug('Layering final convolutional layer %s.' % n_conv)
        if 0 in self.__nets__:
            if pretrain:
                self.AELayers.append(
                    layers.Conv1DLayer(self.AELayers[-1], num_filters=n_1Dfilters[-1], filter_size=filter_size,
                                       nonlinearity=self._nonlinearity))
                self.Conv1DLayers.append(
                    layers.Conv1DLayer(self.Conv1DLayers[-1], num_filters=n_1Dfilters[-1], filter_size=filter_size,
                                       W=self.AELayers[-1].W, b=self.AELayers[-1].b,
                                       nonlinearity=self._nonlinearity))
            else:
                self.Conv1DLayers.append(
                    layers.Conv1DLayer(self.Conv1DLayers[-1], num_filters=n_1Dfilters[-1], filter_size=filter_size,
                                       nonlinearity=self._nonlinearity))
            self.layer_params["Conv1D"].append(
                {"type": "convolutional", "num_filters": self.Conv1DLayers[-1].num_filters,
                 "filter_size": self.Conv1DLayers[-1].filter_size,
                 "shape": layers.get_output_shape(self.Conv1DLayers[-1]), "stride": self.Conv1DLayers[-1].stride})
            if 2 in self.__nets__:
                self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout),
                                                     num_units=layers.get_output_shape(self.Conv1DLayers[-1])[-1] *
                                                               n_1Dfilters[-1],
                                                     nonlinearity=self._nonlinearity)
                self.layer_params["Dense"].append(
                    {"type": "dense", "num_units": self.DenseLayers.num_units,
                     "shape": layers.get_output_shape(self.DenseLayers)})
        if 3 in self.__nets__:
            if pretrain:
                self.AE2DLayers.append(
                    layers.Conv2DLayer(self.AE2DLayers[-1], num_filters=n_2Dfilters[-1], filter_size=filter_size,
                                       nonlinearity=self._nonlinearity))
                self.Conv2DLayers.append(
                    layers.Conv2DLayer(self.Conv2DLayers[-1], num_filters=n_2Dfilters[-1], filter_size=filter_size,
                                       W=self.AE2DLayers[-1].W, b=self.AE2DLayers[-1].b,
                                       nonlinearity=self._nonlinearity))
            else:
                self.Conv2DLayers.append(
                    layers.Conv2DLayer(self.Conv2DLayers[-1], num_filters=n_2Dfilters[-1], filter_size=filter_size,
                                       nonlinearity=self._nonlinearity))
            self.layer_params["Conv2D"].append(
                {"type": "convolutional", "num_filters": self.Conv2DLayers[-1].num_filters,
                 "filter_size": self.Conv2DLayers[-1].filter_size,
                 "shape": layers.get_output_shape(self.Conv2DLayers[-1]), "stride": self.Conv2DLayers[-1].stride})
        if 1 in self.__nets__:
            self.DConvLayers = layers.Conv1DLayer(self.DConvLayers, num_filters=n_1Dfilters[-1],
                                                  filter_size=filter_size,
                                                  nonlinearity=self._nonlinearity)
        logging.debug('Output shapes after pooling layer %s of %s' % (c + 1, n_conv))
        if 0 in self.__nets__:
            logging.debug("1D Conv: " + str(layers.get_output_shape(self.Conv1DLayers[-1])))
            if 2 in self.__nets__:
                logging.debug("MLP: " + str(layers.get_output_shape(self.DenseLayers)))
        if 3 in self.__nets__:
            logging.debug("2D Conv: " + str(layers.get_output_shape(self.Conv2DLayers[-1])))
        if verbose:
            print 'Output shapes after pooling layer %s of %s' % (c, n_conv)
            if 0 in self.__nets__:
                logging
                print "1D Conv: ", layers.get_output_shape(self.Conv1DLayers[-1])
                if 2 in self.__nets__:
                    print "MLP: ", layers.get_output_shape(self.DenseLayers)
            if 3 in self.__nets__:
                print "2D Conv: ", layers.get_output_shape(self.Conv2DLayers[-1])
            print

        """
        Last Max Pooling Layers
        """
        logging.debug("Layering final max pooling layer")
        if pretrain:
            if 0 in self.__nets__:
                self.AELayers.append(layers.MaxPool1DLayer(self.AELayers[-1], pool_size=pool_size))
            if 3 in self.__nets__:
                self.AE2DLayers.append(layers.MaxPool2DLayer(self.AE2DLayers[-1], pool_size=pool_size))
        if 0 in self.__nets__:
            # self.Conv1DLayers.append(layers.MaxPool1DLayer(self.Conv1DLayers[-1], pool_size=pool_size))
            self.Conv1DLayers.append(
                layers.Pool1DLayer(self.Conv1DLayers[-1], pool_size=pool_size, mode='average_exc_pad'))
            self.layer_params["Conv1D"].append(
                {"type": "max pooling", "pool_size": self.Conv1DLayers[-1].pool_size,
                 "stride": self.Conv1DLayers[-1].stride,
                 "shape": layers.get_output_shape(self.Conv1DLayers[-1])})
            if 2 in self.__nets__:
                self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout),
                                                     num_units=layers.get_output_shape(self.Conv1DLayers[-1])[
                                                                   -1] * self._n_1Dfilters[-1],
                                                     nonlinearity=self._nonlinearity)
                self.layer_params["Dense"].append(
                    {"type": "dense", "num_units": self.DenseLayers.num_units,
                     "shape": layers.get_output_shape(self.DenseLayers)})
        if 1 in self.__nets__:
            self.DConvLayers = layers.MaxPool1DLayer(self.DConvLayers, pool_size=pool_size)
        if 3 in self.__nets__:
            self.Conv2DLayers.append(layers.MaxPool2DLayer(self.Conv2DLayers[-1], pool_size=pool_size))
            self.layer_params["Conv2D"].append(
                {"type": "max pooling", "pool_size": self.Conv2DLayers[-1].pool_size,
                 "stride": self.Conv2DLayers[-1].stride,
                 "shape": layers.get_output_shape(self.Conv2DLayers[-1])})
        logging.debug('Output shapes after pooling')
        if 0 in self.__nets__:
            logging.debug("1D Conv: " + str(layers.get_output_shape(self.Conv1DLayers[-1])))
            if 2 in self.__nets__:
                logging.debug("MLP: " + str(layers.get_output_shape(self.DenseLayers)))
        if 3 in self.__nets__:
            logging.debug("2D Conv: " + str(layers.get_output_shape(self.Conv2DLayers[-1])))
        if verbose:
            print 'Output shapes after pooling layer %s of %s' % (c + 1, n_conv)
            if 0 in self.__nets__:
                print "1D Conv: ", layers.get_output_shape(self.Conv1DLayers[-1])
                if 2 in self.__nets__:
                    print "MLP: ", layers.get_output_shape(self.DenseLayers)
            if 3 in self.__nets__:
                print "2D Conv: ", layers.get_output_shape(self.Conv2DLayers[-1])
            print

        if pretrain:
            # Add Decoding Layers
            logging.debug('Adding decoding layers')
            down = len(self.AELayers)
            for i in range(down - 1):
                if 0 in self.__nets__:
                    self.AELayers.append(layers.InverseLayer(self.AELayers[-1], self.AELayers[down - 1 - i]))
                if 3 in self.__nets__:
                    self.AE2DLayers.append(layers.InverseLayer(self.AE2DLayers[-1], self.AE2DLayers[down - 1 - i]))
            if freeze_autoencoder:
                logging.debug("Freezing weights and biases")
                if 0 in self.__nets__:
                    self.Conv1DLayers[-1].params[self.Conv1DLayers[-1].W].remove("trainable")
                    self.Conv1DLayers[-1].params[self.Conv1DLayers[-1].b].remove("trainable")
                if 3 in self.__nets__:
                    self.Conv2DLayers[-1].params[self.Conv2DLayers[-1].W].remove("trainable")
                    self.Conv2DLayers[-1].params[self.Conv2DLayers[-1].b].remove("trainable")
        ###########################################################################################################

        # Dense Layer 1
        logging.debug("Layering dense layer 0 of %s" % (n_dense))
        if 0 in self.__nets__:
            self.Conv1DLayers = layers.DenseLayer(layers.dropout(self.Conv1DLayers[-1], p=dropout),
                                                  num_units=np.prod(
                                                      layers.get_output_shape(self.Conv1DLayers[-1])[-2:]) / 2,
                                                  nonlinearity=self._nonlinearity)
            self.layer_params["Conv1D"].append(
                {"type": "dense", "num_units": self.Conv1DLayers.num_units,
                 "shape": layers.get_output_shape(self.Conv1DLayers)})
            if 2 in self.__nets__:
                self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout),
                                                     num_units=layers.get_output_shape(self.DenseLayers)[-1] / 2,
                                                     nonlinearity=self._nonlinearity)
                self.layer_params["Dense"].append(
                    {"type": "dense", "num_units": self.DenseLayers.num_units,
                     "shape": layers.get_output_shape(self.DenseLayers)})
        if 1 in self.__nets__:
            self.DConvLayers = layers.DenseLayer(layers.dropout(self.DConvLayers, p=dropout),
                                                 num_units=np.prod(layers.get_output_shape(self.DConvLayers)[-2:]) / 2,
                                                 nonlinearity=self._nonlinearity)
        if 3 in self.__nets__:
            self.Conv2DLayers = layers.DenseLayer(layers.dropout(self.Conv2DLayers[-1], p=dropout),
                                                  num_units=np.prod(
                                                      layers.get_output_shape(self.Conv2DLayers[-1])[-3:]) / 2,
                                                  nonlinearity=self._nonlinearity)
            self.layer_params["Conv2D"].append(
                {"type": "dense", "num_units": self.Conv2DLayers.num_units,
                 "shape": layers.get_output_shape(self.Conv2DLayers)})
        logging.debug('Output shapes after dense layer 0')
        if 0 in self.__nets__:
            logging.debug("1D Conv: " + str(layers.get_output_shape(self.Conv1DLayers)))
            if 2 in self.__nets__:
                logging.debug("MLP: " + str(layers.get_output_shape(self.DenseLayers)))
        if 3 in self.__nets__:
            logging.debug("2D Conv: " + str(layers.get_output_shape(self.Conv2DLayers)))
        if verbose:
            print "Output shape after dense layer %s of %s" % (0, n_dense)
            if 0 in self.__nets__:
                print "1D Conv: ", layers.get_output_shape(self.Conv1DLayers)
                if 2 in self.__nets__:
                    print "MLP: ", layers.get_output_shape(self.DenseLayers)
            if 3 in self.__nets__:
                print "2D Conv: ", layers.get_output_shape(self.Conv2DLayers)
            print
        for d in range(n_dense - 2):
            logging.debug('Layering dense layer %s of %s' % (d + 1, n_dense))
            if 0 in self.__nets__:
                self.Conv1DLayers = layers.DenseLayer(layers.dropout(self.Conv1DLayers, p=dropout),
                                                      num_units=layers.get_output_shape(self.Conv1DLayers)[-1] / 2,
                                                      nonlinearity=self._nonlinearity)
                self.layer_params["Conv1D"].append(
                    {"type": "dense", "num_units": self.Conv1DLayers.num_units,
                     "shape": layers.get_output_shape(self.Conv1DLayers)})
                if 2 in self.__nets__:
                    self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout),
                                                         num_units=layers.get_output_shape(self.DenseLayers)[
                                                                       -1] / 2, nonlinearity=self._nonlinearity)
                    self.layer_params["Dense"].append(
                        {"type": "dense", "num_units": self.DenseLayers.num_units,
                         "shape": layers.get_output_shape(self.DenseLayers)})
            if 1 in self.__nets__:
                self.DConvLayers = layers.DenseLayer(layers.dropout(self.DConvLayers, p=dropout),
                                                     num_units=layers.get_output_shape(self.DConvLayers)[-1] / 2,
                                                     nonlinearity=self._nonlinearity)
            if 3 in self.__nets__:
                self.Conv2DLayers = layers.DenseLayer(layers.dropout(self.Conv2DLayers, p=dropout),
                                                      num_units=layers.get_output_shape(self.Conv2DLayers)[-1] / 2,
                                                      nonlinearity=self._nonlinearity)
                self.layer_params["Conv2D"].append(
                    {"type": "dense", "num_units": self.Conv2DLayers.num_units,
                     "shape": layers.get_output_shape(self.Conv2DLayers)})
            logging.debug('Output shapes after dense layer %s of %s' % (d + 1, n_dense))
            if 0 in self.__nets__:
                logging.debug("1D Conv: " + str(layers.get_output_shape(self.Conv1DLayers)))
                if 2 in self.__nets__:
                    logging.debug("MLP: " + str(layers.get_output_shape(self.DenseLayers)))
            if 3 in self.__nets__:
                logging.debug("2D Conv: " + str(layers.get_output_shape(self.Conv2DLayers)))
            if verbose:
                print "Output shape after dense layer %s of %s" % (d + 1, n_dense)
                if 0 in self.__nets__:
                    print "1D Conv: ", layers.get_output_shape(self.Conv1DLayers)
                    if 2 in self.__nets__:
                        print "MLP: ", layers.get_output_shape(self.DenseLayers)
                if 3 in self.__nets__:
                    print "2D Conv: ", layers.get_output_shape(self.Conv2DLayers)
                print

        """
        Output Layer
        """
        logging.debug('Layering output layer.')
        if 0 in self.__nets__:
            self.Conv1DLayers = layers.DenseLayer(layers.dropout(self.Conv1DLayers, p=dropout), num_units=2,
                                                  nonlinearity=self._nonlinearity)
            self.layer_params["Conv1D"].append(
                {"type": "dense", "num_units": self.Conv1DLayers.num_units,
                 "shape": layers.get_output_shape(self.Conv1DLayers)})
            if 2 in self.__nets__:
                self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout), num_units=2,
                                                     nonlinearity=self._nonlinearity)
                self.layer_params["Dense"].append(
                    {"type": "dense", "num_units": self.DenseLayers.num_units,
                     "shape": layers.get_output_shape(self.DenseLayers)})
        if 1 in self.__nets__:
            self.DConvLayers = layers.DenseLayer(layers.dropout(self.DConvLayers, p=dropout), num_units=2,
                                                 nonlinearity=self._nonlinearity)
        if 3 in self.__nets__:
            self.Conv2DLayers = layers.DenseLayer(layers.dropout(self.Conv2DLayers, p=dropout), num_units=2,
                                                  nonlinearity=self._nonlinearity)
            self.layer_params["Conv2D"].append(
                {"type": "dense", "num_units": self.Conv2DLayers.num_units,
                 "shape": layers.get_output_shape(self.Conv2DLayers)})
        logging.debug('Output shapes of final dense layer')
        if 0 in self.__nets__:
            logging.debug("1D Conv: " + str(layers.get_output_shape(self.Conv1DLayers)))
            if 2 in self.__nets__:
                logging.debug("MLP: " + str(layers.get_output_shape(self.DenseLayers)))
        if 3 in self.__nets__:
            logging.debug("2D Conv: " + str(layers.get_output_shape(self.Conv2DLayers)))
        if verbose:
            print('Output shapes of final dense layer')
            if 0 in self.__nets__:
                print "1D Conv: ", layers.get_output_shape(self.Conv1DLayers)
                if 2 in self.__nets__:
                    print "MLP: ", layers.get_output_shape(self.DenseLayers)
            if 3 in self.__nets__:
                print "2D Conv: ", layers.get_output_shape(self.Conv2DLayers)
            print "Done"
        print

    def train_network(self, X_train=None, X_train2D=None, y_train=None, X_val=None, X_val2D=None, y_val=None,
                      num_epochs=1000, stop_err=0.01, d_stable=1e-6, n_stable=10, learning_rate=0.001, beta_1=0.9,
                      beta_2=0.999, epsilon=10e-8, pretrain=True, plotting=False, save_plots=False, **kwargs):
        """
        A function to  be called when the network is ready to be trained.
        :param num_epochs:  int
                            The number of epochs to train the network for. See stop_err, d_stable and n_stable for
                            further details.
        :param stop_err:    float
                            The stop condition specifying the acceptable level of error. Training will exit if this
                            error is achieved.
        :param d_stable:    float
                            A stop condition specifying the level of error deviation necessary for convergence. If the
                            deviation in error stays below this number for more than n_stable epochs, training will
                            cease.
        :param n_stable:    int
                            A stop condition specifying the number of epochs for which the error deviation must stay
                            below d_stable before exiting training, implying either saturation or convergence.
        :param learning_rate:   float
                            The learning rate to be used in the Adam algorithm to train the network weights and biases.
        :param beta_1:      float
                            The exponential decay rate of the first moment estimator of the gradient in the Adam
                            algorithm.
        :param beta_2:      float
                            The exponential decay rate of the second moment estimator of the gradient in the Adam
                            algorithm.
        :param epsilon:     float
                            A small positive number used by Adam to avoid numerical error and division by 0.
        :param pretrain:    Bool
                            whether or not to pretrain the covolutional layers
        :param plotting:    Bool
                            Whether to plot the training and validation errors of each network after training has
                            finished using matplotlib.pyplot module.
        :param save_plots   Bool
                            Whether to save the plots to a file using matplotlib.pyplot module.
        :param kwargs:      keywords
                            Some common keywords:
                                    verbose: Bool
                                        Whether to print information to the screen.
                                    verbosity: int
                                        Verbosity level of the printed information. Useless unless 'verbose' is True.
        :return: nets:      dict,
                            A dictionary containing the various networks which were trained.
        """

        logfile = os.path.join(self.logdir, "train_network.log")
        logging.basicConfig(filename=logfile, level=logging.DEBUG)
        logging.debug('Setting up training variables')

        if X_train is None:
            logging.debug("No 'X_train' data specified... loading pickled data.")
            X_train = cPickle.load(open(os.path.join(self.datadir, "X_train.pkl"), "rb"))
        if X_train2D is None:
            logging.debug("No 'X_train2D' data specified... loading pickled data.")
            X_train2D = cPickle.load(open(os.path.join(self.datadir, "X_train2D.pkl"), "rb"))
        if y_train is None:
            logging.debug("No 'y_train' data specified... loading pickled data.")
            y_train = cPickle.load(open(os.path.join(self.datadir, "y_train.pkl"), "rb"))
        if X_val is None:
            logging.debug("No 'X_val' data specified... loading pickled data.")
            X_val = cPickle.load(open(os.path.join(self.datadir, "X_val.pkl"), "rb"))
        if X_val2D is None:
            logging.debug("No 'X_val2D' data specified... loading pickled data.")
            X_val2D = cPickle.load(open(os.path.join(self.datadir, "X_val2D.pkl"), "rb"))
        if y_val is None:
            logging.debug("No 'y_val' data specified... loading pickled data.")
            y_val = cPickle.load(open(os.path.join(self.datadir, "y_val.pkl"), "rb"))

        verbose = kwargs.get('verbose', False)
        verbosity = kwargs.get('verbosity', 3)

        errorfile = os.path.join(self.statdir, "training_errors.csv")
        timefile = os.path.join(self.statdir, "training_times.csv")
        statfile = os.path.join(self.statdir, "training_stats.csv")
        paramfile = os.path.join(self.statdir, "params.csv")
        num_epochs = int(num_epochs)
        # num_epochs = int(10)
        self.d_stable = d_stable
        self.n_stable = n_stable
        errors = defaultdict(list)
        val_errors = defaultdict(list)
        times = defaultdict(list)
        flags = dict()
        if plotting or save_plots:
            from textwrap import fill
        epochs = dict()
        Total_time = time.time()
        if pretrain:
            if 0 in self.__nets__:
                # Loss expression
                logging.debug('Creating loss expression for 1D autoencoder')
                if verbose and verbosity >= 3:
                    print "Creating loss expression for 1D autoencoder"
                AE1DPred = layers.get_output(self.AELayers[-1])
                AE1DLoss = lasagne.objectives.squared_error(AE1DPred, self.AE_target_var)
                AE1DLoss = AE1DLoss.mean()

                # Parameters and updates
                logging.debug('Getting parameters for 1D autoencoder')
                if verbose and verbosity >= 3:
                    print('Getting parameters for 1D autoencoder')
                AE1DParams = layers.get_all_params(self.AELayers[-1], trainable=True)

                logging.debug('Creating update rules for 1D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating update rules for 1D autoencoder')
                AE1DUpdates = lasagne.updates.adam(AE1DLoss, AE1DParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon,
                                                   learning_rate=learning_rate)

                # Create a loss expression for validation/testing
                logging.debug('Creating validation expression for 1D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating validation expression for 1D autoencoder')
                AE1DTest_pred = layers.get_output(self.AELayers[-1], deterministic=True)
                AE1DTest_loss = lasagne.objectives.squared_error(AE1DTest_pred, self.AE_target_var)
                AE1DTest_loss = AE1DTest_loss.mean()

                # Compile a second function computing the validation loss:
                logging.debug('Creating validation function for 1D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating validation function for 1D autoencoder')
                AE1DVal_fn = theano.function([self.input_var, self.AE_target_var], AE1DTest_loss)

                # Create a training function
                logging.debug('Creating training function for 1D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating training function for 1D autoencoder')
                AE1DTrainer = theano.function([self.input_var, self.AE_target_var], AE1DLoss, updates=AE1DUpdates)
                flags.update({"AE1D": False})
            else:
                # Will cause the pretraining loop to skip training the 1D autoencoder
                flags.update({"AE1D": True})

            if 3 in self.__nets__:
                # Loss expression
                logging.debug('Creating loss expression for 2D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating loss expression for 2D autoencoder')
                AE2DPred = layers.get_output(self.AE2DLayers[-1])
                AE2DLoss = lasagne.objectives.squared_error(AE2DPred, self.AE_target_var2D)
                AE2DLoss = AE2DLoss.mean()

                # Parameters and updates
                logging.debug('Getting parameters for 2D autoencoder')
                if verbose and verbosity >= 3:
                    print('Getting parameters for 2D autoencoder')
                AE2DParams = layers.get_all_params(self.AE2DLayers[-1], trainable=True)

                logging.debug('Creating update rules for 2D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating update rules for 2D autoencoder')
                AE2DUpdates = lasagne.updates.adam(AE2DLoss, AE2DParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon,
                                                   learning_rate=learning_rate)

                # Create a loss expression for validation/testing
                logging.debug('Creating validation expression for 2D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating test for 2D autoencoder')
                AE2DTest_pred = layers.get_output(self.AE2DLayers[-1], deterministic=True)
                AE2DTest_loss = lasagne.objectives.squared_error(AE2DTest_pred, self.AE_target_var2D)
                AE2DTest_loss = AE2DTest_loss.mean()

                # Compile a second function computing the validation loss:
                logging.debug('Creating validation function for 2D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating validation funciton for 2D autoencoder')
                AE2DVal_fn = theano.function([self.input_var2D, self.AE_target_var2D], AE2DTest_loss)

                # Create a training function
                logging.debug('Creating training function for 2D autoencoder')
                if verbose and verbosity >= 3:
                    print('Creating training for 2D autoencoder')
                AE2DTrainer = theano.function([self.input_var2D, self.AE_target_var2D], AE2DLoss, updates=AE2DUpdates)

                flags.update({"AE2D": False})
            else:
                flags.update({"AE2D": True})
            AETrain_time = time.time()
            for i in range(num_epochs):
                logging.debug("Pretraining Epoch %r" % i)
                if verbose and verbosity >= 2:
                    print "##" * 50
                    print "Pretraining Epoch %r" % i
                    print "--" * 50
                if not flags["AE1D"]:
                    logging.debug("Training 1D autoencoder...")
                    if verbose and verbosity >= 2:
                        print "Training 1D autoencoder..."
                    t = time.time()
                    errors["AE1D"].append(AE1DTrainer(X_train, X_train))
                    times["AE1D"].append(time.time() - t)
                    logging.debug('Time: %s' % (times["AE1D"][-1]))
                    logging.debug('Error: %s' % (errors["AE1D"][-1]))
                    val_errors["AE1D"].append(AE1DVal_fn(X_val, X_val))
                    if errors["AE1D"][-1] < stop_err:
                        logging.debug("1D autoencoder training converged.")
                        if verbose and verbosity >= 1:
                            print "1-D training converged."
                        epochs.update({"AE1D": i + 1})
                        flags.update({"AE1D": True})
                    elif self.stablecheck(errors["AE1D"]):
                        logging.debug("1D autoencoder error converged.")
                        if verbose and verbosity >= 1:
                            print "1-D error converged before training completed."
                        epochs.update({"AE1D": i + 1})
                        flags.update({"AE1D": True})
                if not flags["AE2D"]:
                    logging.debug('Training 2D autoencoder...')
                    if verbose and verbosity >= 2:
                        print "Training 2D autoencoder..."
                    t = time.time()
                    errors["AE2D"].append(AE2DTrainer(X_train2D, X_train2D))
                    times["AE2D"].append(time.time() - t)
                    logging.debug('Time: %s' % (times["AE2D"][-1]))
                    logging.debug('Error: %s' % (errors["AE2D"][-1]))
                    val_errors["AE2D"].append(AE2DVal_fn(X_val2D, X_val2D))
                    if errors["AE2D"][-1] < stop_err:
                        logging.debug('2D training converged.')
                        if verbose and verbosity >= 1:
                            print "2-D training converged."
                        epochs.update({"AE2D": i + 1})
                        flags.update({"AE2D": True})
                    elif self.stablecheck(errors["AE2D"]):
                        logging.debug('2D error converged')
                        if verbose and verbosity >= 1:
                            print "2-D error converged before training completed."
                        epochs.update({"AE2D": i + 1})
                        flags.update({"AE2D": True})
                if all(flags.values()):
                    logging.debug("Exiting pretraining after $s epochs. Target epochs: %s" % (i, num_epochs))
                    break
                if verbose:
                    print

            AETrain_time = time.time() - AETrain_time
            if not flags["AE1D"]:
                flags.update({"AE1D": True})
                epochs.update({"AE1D": i + 1})
            if not flags["AE2D"]:
                flags.update({"AE2D": True})
                epochs.update({"AE2D": i + 1})

            if verbose and verbosity >= 1:
                logging.debug('Pretraining completed')
                logging.debug('Total time: %s' % (AETrain_time))
                logging.debug('Epochs: %s' % i)
                print "##" * 50
                print "Pretraining Completed"
                print "Total Time: {:.3f}s".format(AETrain_time)
                print "--" * 50
                print "1-D Autoencoder Error: ", errors["AE1D"][-1]
                print "1-D Autoencoder Validation Error: ", val_errors["AE1D"][-1]
                print "Training Time: {:.3f}s".format(np.sum(times["AE1D"]))
                print "Training Epochs: ", epochs["AE1D"]
                print "--" * 50
                print "2-D Autoencoder Error: ", errors["AE2D"][-1]
                print "2-D Autoencoder Validation Error: ", val_errors["AE2D"][-1]
                print "Training Time: {:.3f}s".format(np.sum(times["AE2D"]))
                print "Training Epochs: ", epochs["AE2D"]
                print "##" * 50
                print "##" * 50
                print
                print

        if 0 in self.__nets__:
            # Loss expression
            logging.debug('Creating loss expression for 1D convolutional network')
            Conv1DPred = layers.get_output(self.Conv1DLayers)
            Conv1DLoss = lasagne.objectives.squared_error(Conv1DPred, self.target_var)
            Conv1DLoss = Conv1DLoss.mean()

            # Parameters and updates
            logging.debug('Getting parameters for 1D convolutional network')
            Conv1DParams = layers.get_all_params(self.Conv1DLayers, trainable=True)
            logging.debug('Creating update rule for 1D convolutional network')
            Conv1DUpdates = lasagne.updates.adam(Conv1DLoss, Conv1DParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon,
                                                 learning_rate=learning_rate)
            # Create a loss expression for validation/testing
            logging.debug('Creating loss expression for validating 1D convolutional network')
            Conv1DTest_pred = layers.get_output(self.Conv1DLayers, deterministic=True)
            Conv1DTest_loss = lasagne.objectives.squared_error(Conv1DTest_pred, self.target_var)
            Conv1DTest_loss = Conv1DTest_loss.mean()

            # Compile a second function computing the validation loss:
            logging.debug('Creating validation function for 1D convolutional network')
            Conv1DVal_fn = theano.function([self.input_var, self.target_var], Conv1DTest_loss)

            # Create a training function
            logging.debug('Creating training function for 1D convolutional network')
            Conv1DTrainer = theano.function([self.input_var, self.target_var], Conv1DLoss, updates=Conv1DUpdates)

            # Initialize storage variables
            Conv1DVal_time = []

            flags.update({"Conv1D": False})
        else:
            flags.update({"Conv1D": True})

        if 2 in self.__nets__:
            # Loss expression
            logging.debug('Creating loss expression for fully connected network')
            DensePred = layers.get_output(self.DenseLayers)
            DenseLoss = lasagne.objectives.squared_error(DensePred, self.target_var)
            DenseLoss = DenseLoss.mean()

            # Parameters and updates
            logging.debug('Getting parameters for fully connected network')
            DenseParams = layers.get_all_params(self.DenseLayers, trainable=True)
            logging.debug('Creating update rule for fully connected network')
            DenseUpdates = lasagne.updates.adam(DenseLoss, DenseParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon,
                                                learning_rate=learning_rate)

            # Create a loss expression for validation/testing
            logging.debug('Creating loss expression for validating fully connected network')
            DenseTest_pred = layers.get_output(self.DenseLayers, deterministic=True)
            DenseTest_loss = lasagne.objectives.squared_error(DenseTest_pred, self.target_var)
            DenseTest_loss = DenseTest_loss.mean()

            # Compile a second function computing the validation loss:
            logging.debug('Creating validation function for fully connected network')
            DenseVal_fn = theano.function([self.input_var, self.target_var], DenseTest_loss)
            # Create a training function
            logging.debug('Creating training function for fully connected network')
            DenseTrainer = theano.function([self.input_var, self.target_var], DenseLoss, updates=DenseUpdates)
            # Initialize storage variables
            DenseVal_time = []

            flags.update({"Dense": False})
        else:
            flags.update({"Dense": True})

        if 1 in self.__nets__:
            # Loss expression
            logging.debug('Creating loss expression for non-pretrained 1D convolutional network')
            DConvPred = layers.get_output(self.DConvLayers)
            DConvLoss = lasagne.objectives.squared_error(DConvPred, self.target_var)
            DConvLoss = DConvLoss.mean()

            # Create an update expression for training
            logging.debug('Getting parametersfor non-pretrained 1D convolutional network')
            DConvParams = layers.get_all_params(self.DConvLayers, trainable=True)
            logging.debug('Creating update rule for non-pretrained 1D convolutional network')
            DConvUpdates = lasagne.updates.adam(DConvLoss, DConvParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon,
                                                learning_rate=learning_rate)

            # Create a loss expression for validation/testing
            logging.debug('Creating loss expression for validating non-pretrained 1D convolutional network')
            DConvTest_pred = layers.get_output(self.DConvLayers, deterministic=True)
            DConvTest_loss = lasagne.objectives.squared_error(DConvTest_pred, self.target_var)
            DConvTest_loss = DConvTest_loss.mean()

            # Compile a second function computing the validation loss:
            logging.debug('Creating validation function for non-pretrained 1D convolutional network')
            DConvVal_fn = theano.function([self.input_var, self.target_var], DConvTest_loss)

            # Create a training function
            logging.debug('Creating training function for non-pretrained 1D convolutional network')
            DConvTrainer = theano.function([self.input_var, self.target_var], DConvLoss, updates=DConvUpdates)

            # Initialize Storage Variables
            DConvVal_time = []

            flags.update({"DConv": False})
        else:
            flags.update({"DConv": True})

        if 3 in self.__nets__:
            # Loss expression
            logging.debug('Creating loss expression for 2D convolutional network')
            Conv2DPred = layers.get_output(self.Conv2DLayers)
            Conv2DLoss = lasagne.objectives.squared_error(Conv2DPred, self.target_var)
            Conv2DLoss = Conv2DLoss.mean()

            # Parameters and updates
            logging.debug('Getting parameters for 2D convolutional network')
            Conv2DParams = layers.get_all_params(self.Conv2DLayers, trainable=True)
            Conv2DUpdates = lasagne.updates.adam(Conv2DLoss, Conv2DParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon,
                                                 learning_rate=learning_rate)
            # Create a loss expression for validation/testing
            logging.debug('Creating loss expression for validating 2D convolutional network')
            Conv2DTest_pred = layers.get_output(self.Conv2DLayers, deterministic=True)
            Conv2DTest_loss = lasagne.objectives.squared_error(Conv2DTest_pred, self.target_var)
            Conv2DTest_loss = Conv2DTest_loss.mean()

            # Compile a second function computing the validation loss:
            logging.debug('Creating validation function for 2D convolutional network')
            Conv2DVal_fn = theano.function([self.input_var2D, self.target_var], Conv2DTest_loss)
            logging.debug('Creating training function for 2D convolutional network')
            Conv2DTrainer = theano.function([self.input_var2D, self.target_var], Conv2DLoss, updates=Conv2DUpdates)

            # Initialize Storage Variables
            Conv2DVal_time = []

            flags.update({"Conv2D": False})
        else:
            flags.update({"Conv2D": True})
        NetTrain_time = time.time()
        for i in range(num_epochs):
            logging.debug("Training epoch %r" % i)
            if verbose and verbosity >= 2:
                print "##" * 50
                print "Training Epoch %r" % i
                print "--" * 50
            if not flags["Conv1D"]:
                logging.debug("Training pretrained 1-D convolutional net...")
                if verbose and verbosity >= 2:
                    print "Training pretrained 1-D convolutional net..."
                t = time.time()
                errors["Conv1D"].append(Conv1DTrainer(X_train, y_train))
                times["Conv1D"].append(time.time() - t)
                logging.debug("Time: %s" % times["Conv1D"][-1])
                logging.debug("Error: %s" % errors["Conv1D"][-1])
                t = time.time()
                val_errors["Conv1D"].append(Conv1DVal_fn(X_val, y_val))
                Conv1DVal_time.append(time.time() - t)
                if errors["Conv1D"][-1] < stop_err:
                    logging.debug('1D training converged')
                    if verbose and verbosity >= 1:
                        print "1-D training converged."
                    epochs.update({"Conv1D": i + 1})
                    flags.update({"Conv1D": True})
                elif self.stablecheck(errors["Conv1D"]):
                    logging.debug('1D error converged')
                    if verbose and verbosity >= 1:
                        print "1-D error converged before training completed."
                    epochs.update({"Conv1D": i + 1})
                    flags.update({"Conv1D": True})
            if not flags["Conv2D"]:
                logging.debug("Training pretrained 2D convolutional net")
                if verbose and verbosity >= 2:
                    print "Training pretrained 2-D convolutional net..."
                t = time.time()
                errors["Conv2D"].append(Conv2DTrainer(X_train2D, y_train))
                times["Conv2D"].append(time.time() - t)
                logging.debug("Time: %s" % times["Conv2D"][-1])
                logging.debug("Error: %s" % errors["Conv2D"][-1])
                t = time.time()
                val_errors["Conv2D"].append(Conv2DVal_fn(X_val2D, y_val))
                Conv2DVal_time.append(time.time() - t)
                if errors["Conv2D"][-1] < stop_err:
                    logging.debug("2D training converged")
                    if verbose and verbosity >= 1:
                        print "2-D training converged."
                    epochs.update({"Conv2D": i + 1})
                    flags.update({"Conv2D": True})
                elif self.stablecheck(errors["Conv2D"]):
                    logging.debug("2D error converged")
                    if verbose and verbosity >= 1:
                        print "2-D error converged before training completed."
                    epochs.update({"Conv2D": i + 1})
                    flags.update({"Conv2D": True})
            if not flags["DConv"]:
                logging.debug("Training unpretrained convolutional net")
                if verbose and verbosity >= 2:
                    print "Training Unpretrained Convolutional Net..."
                t = time.time()
                errors["DConv"].append(DConvTrainer(X_train, y_train))
                times["DConv"].append(time.time() - t)
                logging.debug("Time: %s" % times["DConv"][-1])
                logging.debug("Error: %s" % errors["DConv"][-1])
                t = time.time()
                val_errors["DConv"].append(DConvVal_fn(X_val, y_val))
                DConvVal_time.append(time.time() - t)

                if errors["DConv"][-1] < stop_err:
                    logging.debug("1D non-pretrained training converged.")
                    if verbose and verbosity >= 1:
                        print "1-D non-pretrained training converged."
                    epochs.update({"DConv": i + 1})
                    flags.update({"DConv": True})
                elif self.stablecheck(errors["DConv"]):
                    logging.debug("1D non-pretrained error converged.")
                    if verbose and verbosity >= 1:
                        print "1-D non-pretrained error converged before training completed."
                    epochs.update({"DConv": i + 1})
                    flags.update({"DConv": True})
            if not flags["Dense"]:
                logging.debug("Training MLP")
                if verbose and verbosity >= 2:
                    print "Training MLP..."
                t = time.time()
                errors["Dense"].append(DenseTrainer(X_train, y_train))
                times["Dense"].append(time.time() - t)
                logging.debug("Time: %s" % times["Dense"][-1])
                logging.debug("Error: %s" % errors["Dense"][-1])
                t = time.time()
                val_errors["Dense"].append(DenseVal_fn(X_val, y_val))
                DenseVal_time.append(time.time() - t)
                if errors["Dense"][-1] < stop_err:
                    logging.debug("MLP training converged")
                    if verbose and verbosity >= 1:
                        print "MLP training converged."
                    epochs.update({"Dense": i + 1})
                    flags.update({"Dense": True})
                elif self.stablecheck(errors["Dense"]):
                    logging.debug("MLP error converged.")
                    if verbose and verbosity >= 1:
                        print "Fully connected error converged before training completed."
                    epochs.update({"Dense": i + 1})
                    flags.update({"Dense": True})
            if all(flags.values()):
                break
            if verbose:
                print
        Total_time = time.time() - Total_time
        NetTrain_time = time.time() - NetTrain_time
        if 0 in self.__nets__:
            if not flags["Conv1D"]:
                epochs.update({"Conv1D": i + 1})
                flags.update({"Conv1D": True})
        if 3 in self.__nets__:
            if not flags["Conv2D"]:
                epochs.update({"Conv2D": i + 1})
                flags.update({"Conv2D": True})
        if 1 in self.__nets__:
            if not flags["DConv"]:
                epochs.update({"DConv": i + 1})
                flags.update({"DConv": True})
        if 2 in self.__nets__:
            if not flags["Dense"]:
                epochs.update({"Dense": i + 1})
                flags.update({"Dense": True})
        self.epochs = epochs

        logging.debug("All networks trained: %s" % (all(flags.values())))
        logging.debug("Training complete")
        logging.debug("Total time: %s" % NetTrain_time)
        logging.debug("Epochs: %s" % i)
        if verbose and verbosity >= 1:
            print
            print "##" * 50
            print "Training Completed: %r Convolutional Layers, %r Dense Layers" % (self._n_conv, self._n_dense)
            print "Total Time: {:.3f}s".format(NetTrain_time)
            print "%r 1-D filters, %r 2-D filters" % (self._n_1Dfilters[0], self._n_2Dfilters[0])
            print "Nonlinearity: %r" % (self.nonlinearity)
            print "Frozen Weights: %r" % (self._frozen)
            print "--" * 50
            if 0 in self.__nets__:
                print "1-D Training Error: ", errors["Conv1D"][-1]
                print "1-D Mean Validation Error: ", np.mean(errors["Conv1D"][-n_stable:])
                print "Training Time: {:.3f}s".format(np.sum(times["Conv1D"]))
                print "Training Epochs: ", epochs["Conv1D"]
                print "--" * 50
            if 3 in self.__nets__:
                print "2-D Training Error: ", errors["Conv2D"][-1]
                print "2-D Mean Validation Error: ", np.mean(errors["Conv2D"][-n_stable:])
                print "Training Time: {:.3f}s".format(np.sum(times["Conv2D"]))
                print "Training Epochs: ", epochs["Conv2D"]
                print "--" * 50
            if 1 in self.__nets__:
                print "1-D Unpretrained Error: ", errors["DConv"][-1]
                print "1-D Unpretrained Validation Error: ", np.mean(errors["DConv"][-n_stable:])
                print "Training Time: {:.3f}s".format(np.sum(times["DConv"]))
                print "Training Epochs: ", epochs["DConv"]
                print "--" * 50
            if 2 in self.__nets__:
                print "MLP Training Error: ", errors["Dense"][-1]
                print "MLP Mean Validation Error: ", np.mean(errors["Dense"][-n_stable:])
                print "Training Time: {:.3f}s".format(np.sum(times["Dense"]))
                print "Training Epochs: ", epochs["Dense"]
            print "##" * 50
            print "##" * 50
            print "##" * 50

        logging.debug("Writing errors to csv file: %s" % (errorfile))
        # header = ['Epoch']
        header = epochs.keys()
        with open(errorfile, 'wb') as errorfile:
            error_writer = csv.writer(errorfile, quoting=csv.QUOTE_ALL)
            error_writer.writerow(header)
            for i in range(num_epochs):
                row = [i]
                for key in errors.keys():
                    try:
                        row.append(errors[key][i])
                    except IndexError:
                        row.append("")
                error_writer.writerow(row)

        logging.debug("Writing times to csv file: %s" % (timefile))
        with open(timefile, 'wb') as timefile:
            time_writer = csv.writer(timefile, quoting=csv.QUOTE_ALL)
            time_writer.writerow(header)
            for i in range(num_epochs):
                row = [i]
                for key in times.keys():
                    try:
                        row.append(times[key][i])
                    except IndexError:
                        row.append("")
                time_writer.writerow(row)

        logging.debug("Writing stats to csv file: %s" % (statfile))
        statheader = ['Network', 'Mean Error', 'Median Error', 'Error Variance', 'Max Error', 'Min Error', 'Mean Time',
                      'Median Time', 'Time Variance', 'Max Time', 'Min Time', 'Total Time', "Epochs"]
        with open(statfile, 'wb') as statfile:
            stat_writer = csv.writer(statfile, quoting=csv.QUOTE_ALL)
            stat_writer.writerow(statheader)
            for key in errors.keys():
                _error = errors[key]
                _time = times[key]
                row = [key, np.mean(_error), np.median(_error), np.var(_error), np.max(_error), np.min(_error),
                       np.mean(_time), np.median(_time), np.var(_time), np.max(_time), np.min(_time), np.sum(_time),
                       epochs[key]]
                stat_writer.writerow(row)
            if pretrain:
                stat_writer.writerow(['Autoencoder Training Time', AETrain_time])
            stat_writer.writerow(['Network Training Time', NetTrain_time])
            stat_writer.writerow(['Total Time', Total_time])

        with open(paramfile, 'wb') as paramfile:
            param_writer = csv.writer(paramfile, quoting=csv.QUOTE_ALL)
            for key in self.layer_params.keys():
                for l in self.layer_params[key]:
                    row_header = ["Net"]
                    row_header.extend(l.keys())
                    param_writer.writerow(row_header)
                    row = [key]
                    row.extend(l.values())
                    param_writer.writerow(row)

        if plotting or save_plots:
            logging.debug("Generating plots")
            if 0 in self.__nets__ and 2 in self.__nets__:
                plt.title(fill("MLP and 1-D Unpretrained Convolutional Network Training and Validation Error", 45))
                plt.plot(errors["DConv"], 'g', label='Convolutional, Training')
                plt.plot(val_errors["DConv"], 'b', label='Convolutional, Validation')
                plt.plot(errors["Dense"], 'k', label='MLP, Training')
                plt.plot(val_errors["Dense"], 'r', label='MLP, Validation')
                plt.xlabel("Epoch")
                plt.ylabel("Mean Squared Error")
                plt.legend()
                if save_plots:
                    logging.debug("Saving plot of MLP/Convolutional comparison")
                    dir = os.path.dirname(__file__)
                    name = self.__name__ + "__" + str(time.time())
                    figfile = os.path.join(dir, "../plots/" + name + "_MLPconvcompare.png")
                    plt.savefig(figfile)
                if plotting:
                    plt.show()
            if 0 in self.__nets__:
                plt.title(
                    fill("1-D Convolutional Network Training and Validation Error: %r filters" % (self._n_1Dfilters[0]),
                         45))
                plt.plot(errors["Conv1D"], 'g', label='Training')
                plt.plot(val_errors["Conv1D"], 'b', label='Validation')
                plt.xlabel("Epoch")
                plt.ylabel("Mean Squared Error")
                plt.legend()
                if save_plots:
                    logging.debug("Saving plot of 1D convolutional training and validation error")
                    figfile = os.path.join(dir, "../plots/" + name + "_1Dconvnet.png")
                    plt.savefig(figfile)
                if plotting:
                    plt.show()
                if 1 in self.__nets__:
                    plt.title(
                        fill(
                            "1-D Convolutional Network Pretrained Vs Untrained Training and Validation Error: %r filters" % (
                                self._n_1Dfilters[0]), 45))
                    plt.plot(errors["Conv1D"], 'g', label='Training')
                    plt.plot(val_errors["Conv1D"], 'b', label='Validation')
                    plt.plot(errors["DConv"], 'k', label='Training')
                    plt.plot(val_errors["DConv"], 'r', label='Validation')
                    plt.xlabel("Epoch")
                    plt.ylabel("Mean Squared Error")
                    plt.legend()
                    if save_plots:
                        logging.debug("Saving plot of 1D convolutional training and validation error")
                        figfile = os.path.join(dir, "../plots/" + name + "_1DPreTvsUnPreTconvnet.png")
                        plt.savefig(figfile)
                    if plotting:
                        plt.show()

            if 3 in self.__nets__:
                plt.title(
                    fill("2-D Convolutional Network Training and Validation Error: %r filters" % (self._n_2Dfilters[0]),
                         45))
                plt.plot(errors["Conv2D"], 'g', label='Training')
                plt.plot(val_errors["Conv2D"], 'b', label='Validation')
                plt.xlabel("Epoch")
                plt.ylabel("Mean Squared Error")
                plt.legend()
                if save_plots:
                    figfile = os.path.join(dir, "../plots/" + name + "_2Dconvnet.png")
                    plt.savefig(figfile)
                if plotting:
                    plt.show()

        # Add the trained networks to self.
        self.nets = dict()

        if 0 in self.__nets__:
            self.nets.update({0: theano.function([self.input_var],
                                                 layers.get_output(self.Conv1DLayers, deterministic=True),
                                                 name="1D Convolutional Network")})
        if 1 in self.__nets__:
            self.nets.update({1: theano.function([self.input_var],
                                                 layers.get_output(self.DConvLayers, deterministic=True),
                                                 name="1D Convolutional Network Sans Pretraining")})
        if 2 in self.__nets__:
            self.nets.update({2: theano.function([self.input_var],
                                                 layers.get_output(self.DenseLayers, deterministic=True),
                                                 name="MLP Network")})
        if 3 in self.__nets__:
            self.nets.update({3: theano.function([self.input_var2D],
                                                 layers.get_output(self.Conv2DLayers, deterministic=True),
                                                 name="2D Convolutional Network")})

        data1, data2, target = X_train[:20], X_train2D[:20], y_train[:20]
        # print "Test"
        # print "Expected: ", target
        predictions, networks, coordinates, errors = self.solve(input1D=self.shape_data1D(data1D=data1, batched=True, batch_axis=0),
            input2D=self.shape_data2D(data2, channel_axis=1, batch_axis=0, batched=True), targets=target, network_ids=3)
        # print "Predictions: ", predictions
        # print "Networks: ", networks
        # print "Errors: ", errors
        # print "Coordinates"
        # print coordinates[0]
        # print coordinates
        # print
        # example = rgb_to_grey(data2[0])
        # plt.imshow(example, alpha=0.5)
        # overlay = rotate(example, 180. * coordinates[0][2] / np.pi, reshape=False)
        # overlay = zoom(overlay, coordinates[0][3])
        # overlay = shift(overlay, coordinates[0][:2])
        # # example = np.zeros((64, 64, 3), dtype=float)
        # # example[:, :, 1] = overlay
        # plt.imshow(overlay, alpha=0.5)
        # plt.show()
        return self.nets

    def stablecheck(self, error_list):
        """
        Check to see if the error has stabilized, i.e. the change in error has stayed below self.d_stable for at
        self.n_stable iterations
        :param error_list: list of floats
                            the errors to be checked for stability
        :return: Bool
                            whether or not the deviation in error has stabilized
        """
        diff = np.diff(error_list)
        diff = np.abs(diff[-self.n_stable:]) < self.d_stable
        if len(diff) < self.n_stable:
            return False
        else:
            return diff.all()

    def split_data(self, data1D=None, data2D=None, targets=None, train_prop=0.75, shuffle=True, verbose=False,
                   **kwargs):
        """
        Splits the data into train_prop% training and 1-train_prop% validation data. Stores the split data in the data directory.
        :param data1D:      3D array of floats, optional
                            A 3D array containing the curvature data to be split. Dimensions are (batch size, 1, curvature values). If not specified, loads data from the data directory
        :param data2D:      4D array of floats, optional
                            A 4D array containing the image data to be split. Dimensions are (batch size, no. channels, no. rows, no.cols). If not specified, loads data from the data directory
        :param targets:     array-like, optional
                            A 2D array-like object containing the targets for each data point. If not specified, loads data from the data directory.
        :param train_prop:  float, optional,
                            Proportion of data to be used for training. The remaining is used as validation data
        :param shuffle:     Bool, optional,
                            Whether or not to shuffle the data before splitting it. This is recommended
        :param verbose:     Bool, optional,
                            Whether to print output to the screen
        :return X_train, X_train2D, y_train, X_val, X_val2D, y_val
                            X_train, X_train2D:     ndarrays
                            The training input data for the 1D and 2D networks respectively
                            X_val, X_val2D:     ndarrays
                            The validation input data for the 1D and 2D networks respectively
                            y_train, y_val:     ndarrays
                            The training target data for the training and validation respectively
        """
        if verbose:
            print "Splitting Data..."
        if data1D is None:
            data1D = cPickle.load(open(os.path.join(self.datadir, "data1D.pkl", "rb")))
        if data2D is None:
            data2D = cPickle.load(open(os.path.join(self.datadir, "data2D.pkl", "rb")))
        if targets is None:
            targets = cPickle.load(open(os.path.join(self.datadir, "targets.pkl", "rb")))
        indices = np.arange(len(data1D))
        inx = int(train_prop * len(data1D))

        if shuffle:
            np.random.shuffle(indices)

        train_excerpt = indices[:inx]
        val_excerpt = indices[inx:]

        X_train, y_train = data1D[train_excerpt], targets[train_excerpt]
        X_train2D = data2D[train_excerpt]
        X_val, y_val = data1D[val_excerpt], targets[val_excerpt]
        X_val2D = data2D[val_excerpt]

        cPickle.dump(X_train, open(os.path.join(self.datadir, "X_train.pkl"), "wb"))
        cPickle.dump(X_train2D, open(os.path.join(self.datadir, "X_train2D.pkl"), "wb"))
        cPickle.dump(y_train, open(os.path.join(self.datadir, "y_train.pkl"), "wb"))

        cPickle.dump(X_val, open(os.path.join(self.datadir, "X_val.pkl"), "wb"))
        cPickle.dump(X_val2D, open(os.path.join(self.datadir, "X_val2D.pkl"), "wb"))
        cPickle.dump(y_val, open(os.path.join(self.datadir, "y_val.pkl"), "wb"))

        if verbose:
            print "Done"
            print
        return X_train, X_train2D, y_train, X_val, X_val2D, y_val

    def shape_data2D(self, data2D=None, channel_axis=-1, batched=False, batch_axis=0):
        """
        Shapes data to proper dimensions by rearranging channel dimension, batch dimension, etc. Also dumps data to a data file for later use.
        :param data2D: ndarray, optional
                            Image data to be shaped. If the pixels are ints, scale them to floats in [0, 1]. If not specified, loads data from the data directory
        :param channel_axis: int, optional
                            Axis which stores the channel width. By default this value is -1 in the case of the last
                            axis denoting the number of channels. Also accepts None if channel axis is to be inferred from smallest dimension
        :param batched: Bool, optional
                            Whether or not the incoming data is a singleton or a batch. If incoming data has 4
                            dimensions it is assumed to be batched.
        :param batch_axis: int, optional
                            Integer specifying which axis contains the batch size. Also accepts None if batch axis is to be inferred from largest dimension. Specifying None also sets batched to True. Be careful when working with small data sets
        :return: ndarray
                            Reshaped data whose dimensions are (batch size, number of channels, height, width)
        """
        if data2D is None:
            data2D = cPickle.load(open(os.path.join(self.datadir, "data2D.pkl"), "rb"))

        assert len(data2D.shape) <= 4, "Image data has too many dimensions. Unsure how to reshape"
        if data2D.dtype == int:
            data2D = data2D / 255.
        elif data2D.dtype == float:
            pass
        else:
            raise TypeError("Image data type must be float or int. Instead got %r" % (data2D.dtype))

        if channel_axis is None:
            channel_axis = np.argmin(data2D.shape)
        if batch_axis is None:
            batched = True
            batch_axis = np.argmax(data2D.shape)

        if len(data2D.shape) is 2:
            reshaped = np.zeros((1, 1, data2D.shape[0], data2D.shape[1]), dtype=float)
            reshaped[0, 0, :, :] = data2D
        elif len(data2D.shape) is 3:
            if not batched:
                axes = []
                for i in range(3):
                    if i is not (channel_axis % 3):
                        axes.append(i)
                data2D.reshape((data2D.shape[channel_axis], data2D.shape[axes[0]], data2D.shape[axes[1]]))
                reshaped = np.zeros((1, data2D.shape[0], data2D.shape[1], data2D.shape[2]), dtype=float)
                reshaped[0, :, :, :] = data2D
            else:
                axes = []
                for i in range(3):
                    if i is not (batch_axis % 3):
                        axes.append(i)
                data2D.reshape((data2D.shape[batch_axis], data2D.shape[axes[0]], data2D.shape[axes[1]]))
                reshaped = np.zeros((data2D.shape[0], 1, data2D.shape[1], data2D.shape[2]), dtype=float)
                reshaped[:, 0, :, :] = data2D
        elif len(data2D.shape) is 4:
            axes = []
            for i in range(4):
                if i is not (channel_axis % 4) and i is not (batch_axis % 4):
                    axes.append(i)
            data2D.reshape(
                (data2D.shape[batch_axis], data2D.shape[channel_axis], data2D.shape[axes[0]], data2D.shape[axes[1]]))
            reshaped = data2D

        m, n = np.max(reshaped), np.min(reshaped)
        if m > 1.:
            raise ValueError("Image pixel values are outside accepted range. Max value %r must be less than 1." % (m))
        elif n < 0.:
            raise ValueError("Image pixel values are outside accepted range. Min value %r must be nonnegative" % (n))
        return reshaped

    def shape_data1D(self, data1D=None, channel_axis=1, batched=False, batch_axis=0):
        """
        Shapes data to proper dimensions by rearranging curvature dimension, batch dimension, etc. Also dumps data to a data file for later use.
        :param data1D: ndarray, optional
                            Curvature data to be reshaped. If not specified, loads data from the data directory
        :param batched: Bool, optional
                            Whether or not the incoming data is a singleton or a batch. If incoming data has 3
                            dimensions it is assumed to be batched.
        :param batch_axis: int, optional
                            Integer specifying which axis contains the batch size. Also accepts None if batch axis is to be inferred from largest dimension. Specifying None also sets batched to True. Be careful when working with small data sets.
        :return: ndarray
                            Reshaped data whose dimensions are (batch size, number of channels, data points)
        """
        if data1D is None:
            data1D = cPickle.load(open(os.path.join(self.datadir, "data1D.pkl"), "rb"))

        assert len(data1D.shape) <= 3, "Data has too many dimensions. Unsure how to reshape."

        if channel_axis is None:
            channel_axis = np.argmin(data1D.shape)

        if batch_axis is None:
            batched = True
            batch_axis = np.argmax(data1D.shape)

        if len(data1D.shape) is 1:
            reshaped = np.zeros((1, 1, data1D.shape[0]), dtype=float)
            reshaped[0, 0, :] = data1D
        elif len(data1D.shape) is 2:
            if not batched:
                for i in range(2):
                    if i is not channel_axis:
                        axes = [i]
                reshaped = np.zeros((1, data1D.shape[channel_axis], data1D.shape[axes[0]]), dtype=float)
                reshaped[0, :, :] = data1D
            else:
                for i in range(2):
                    if i is not (batch_axis % 2):
                        axes = [i]
                reshaped = np.zeros((data1D.shape[batch_axis], 1, data1D.shape[axes[0]]), dtype=float)
                reshaped[:, 0, :] = data1D
        elif len(data1D.shape) is 3:
            axes = []
            for i in range(3):
                if i is not (channel_axis % 3) and i is not (batch_axis % 3):
                    axes.append(i)
            data1D.reshape((data1D.shape[batch_axis], data1D.shape[axes[channel_axis]], data1D.shape[axes[0]]))
            reshaped = data1D
        return reshaped

    def to_output(self, x):
        """
        Helper function to convert normalized predictions from [0, 1] to [1/2, 2] and [0, 2*pi]
        :param x: array-like
                    The predictions for scale, and angle (in that order) normalized to [0, 1].
        :return: scale, angle
                    The un-normalized predictions. Angle is in radians.
        """
        x[0] = x[0] ** 2.
        x[1] = (x[1] + .5) * 2. * np.pi
        x[1] = x[1] % (2. * np.pi)
        return x

    def get_error(self, x, y, relative=False, epsilon=1e-8):
        """
        Helper function to compute the error between the network prediction x and network target y.
        :param x: sequence of floats
                    A sequence containing estimations of y. Expected shape is (2L,)
        :param y: sequence of floats
                    A sequence containing values which x is to be estimating. Expected shape is (2L,)
        :param relative: Bool, optional
                    Denotes whether to use absolute or relative error calculations.
        :param epsilon: float, optional
                    A small positive number to avoid division by 0.
        :return: (err1, err2)
                    if relative error: err = |(x-y-epsilon)/(y+epsilon)|
                    if absolute error: err = |x-y|
        """
        if not relative:
            error = (np.abs(x[0] - y[0]), np.abs(x[1] - y[1]))
        else:
            error = (np.abs(x[0] - y[0] - epsilon) / np.abs(y[0] + epsilon),
                     np.abs(x[1] - y[1] - epsilon) / np.abs(y[1]) + epsilon)
        return error

    def solve(self, input1D=None, input2D=None, targets=None, network_ids=None, timing=False, batch_process=False,
              feedback=True, **kwargs):
        """
        :param input1D: ndarray, optional
                            An array containing the curvature data to be solved. The dimensions must be
                            (batch size, 1, data points per sample) where the number of data points is the same as
                            during training. See shape_data1D for automatic reshaping tools.
        :param input2D:  ndarray, optional
                            An array containing images to be passed to the 2D convolutional network. The dimensions must
                            be (batch size, number of channels, height, width). Number of channels, height, and width
                            must be the same as during training. See shape_data2D for automatic reshaping tools.
        :param targets: array-like, optional
                            An array containing the target values for the input data. Must have same first dimension
                            (batch size) as input1D and/or input2D. If specified, errors are returned as an array with
                            shape (batch size, 2). See get_error for possible keyword arguments.
        :param network_ids: int, sequence of ints, optional
                            Which networks to use for predictions:
                                                0: 1D convolutional network
                                                1: 1D convolutional network w/o pretraining
                                                2: MLP network
                                                3: 2D convolutional network
                            Not specified:      return predictions from all networks.
                            int:                return only the prediction from the network specified by network_id
                            sequence of ints:   return only the predictions from the networks whose id's are specified
        :param timing:  Bool, optional
                            Whether or not to time the solver. If true, then prints the time to the screen depending on
                            verbosity level:
                                    0: Print just the total time
                                    1: No change
                                    2: Print the time for each sample
                                    3: Print timing statistics after completion
        :return: predictions, networks, [errors]
                            predictions:    ndarray with shape (batch size, num network ids, 2) where the last dimension
                                            is the (scale, angle) prediction from the specified networks in the same
                                            order as which they were specified. Input number denotes which input the
                                            predictions correspond to.
                            networks:       The names of the networks used and the order in which they were
                                            used/specified.
                            errors:         If targets were specified, this extra variable whose shape is
                                            (batch size, num network ids, 2) is returned which holds the scale and angle
                                            errors repectively for each sample in the batch and each network specified
                                            by network id.
        """
        verbose = kwargs.get('verbose', False)
        verbosity = kwargs.get('verbosity', 2)
        if targets is not None:
            targets = np.asarray(targets)
            assert targets.shape[
                       -1] == 2, "Targets shape not in output space. Expected 2 targets per sample, instead got $r per sample." % (
                targets.shape[-1])
            relative = kwargs.get("relative", True)
            epsilon = kwargs.get("epsilon", 1e-8)

        if input1D is None and input2D is None:
            raise UnboundLocalError("No input specified.")
        elif input2D is None:
            assert len(
                input1D.shape) == 3, "ShapeError: Improper number of dimensions for 1D input. Expected 3 with dimensions (batch size, 1, data points per sample) and got %r" % (
                input1D.shape)
            assert input1D.shape[
                       -1] == self._input1D_size, "1D ShapeError: Expected %r inputs per sample. Got %r instead." % (
                self._input1D_size, input1D.shape[-1])
            if targets is not None:
                assert input1D.shape[0] == targets.shape[
                    0], "1D input batch size (%r) does not match expected batch size (%r)." % (
                    input1D.shape[0], targets.shape[0])
            batch_size = input1D.shape[0]
        elif input1D is None:
            assert len(
                input2D.shape) == 4, "ShapeError: Improper number of dimensions for image input. Expected 4 with dimensions (batch size, number of channels, height, width) and got %r" % (
                input2D.shape)
            assert input2D.shape[
                   1:] == self._input2D_shape, "Image ShapeError: Expected image input with shape %r, instead got input with shape %r." % (
                self._input2D_shape, input2D.shape[1:])
            if targets is not None:
                assert input2D.shape[0] == targets.shape[
                    0], "Image input batch size (%r) does not match expected batch size (%r)." % (
                    input2D.shape[0], targets.shape[0])
            batch_size = input2D.shape[0]
        else:
            assert len(
                input1D.shape) == 3, "ShapeError: Improper number of dimensions for 1D input. Expected 3 with dimensions (batch size, 1, data points per sample) and got %r" % (
                input1D.shape)
            assert input1D.shape[
                       -1] == self._input1D_size, "1D ShapeError: Expected %r inputs per sample. Got %r instead." % (
                self._input1D_size, input1D.shape[-1])
            assert len(
                input2D.shape) == 4, "ShapeError: Improper number of dimensions for image input. Expected 4 with dimensions (batch size, number of channels, height, width) and got %r" % (
                input2D.shape)
            assert input2D.shape[
                   1:] == self._input2D_shape, "Image ShapeError: Expected image input with shape %r, instead got input with shape %r." % (
                self._input2D_shape, input2D.shape[1:])
            assert input1D.shape[0] == input2D.shape[0], "1-D batch size (%r) and 2-D batch size (%r) do not match." % (
                input1D.shape[0], input2D.shape[0])
            batch_size = input1D.shape[0]
            if targets is not None:
                assert targets.shape[0] == batch_size, "Targets batch size (%r) does not match batch size (%r)." % (
                    targets.shape[0], batch_size)

        if network_ids is None:
            # network_ids = [i for i in range(7)]
            network_ids = [i for i in range(4)]
        elif type(network_ids) is int:
            network_ids = [network_ids]
        elif type(network_ids) is np.ndarray:
            network_ids = list(network_ids.flatten())
        else:
            # should raise an error if network_ids is not iterable, which is the necessary functionality
            iter(network_ids)
        networks = [self.nets[id].name for id in network_ids]
        for i in range(len(network_ids)):
            if i not in range(7):
                raise ValueError(
                    "Invalid network ID supplied. Must be an integer between 0 and 6. The invalid ID is %r" % (i))

        if input1D is None and 0 in network_ids or 6 in network_ids or 5 in network_ids:
            warnings.warn(
                "Asked to return data from 1-D convolutional networks but no 1-D data was given. Ignoring invalid requests.",
                Warning)
            while 0 in network_ids or 6 in network_ids:
                try:
                    network_ids.remove(0)
                    network_ids.remove(6)
                except ValueError:
                    pass
        if input2D is None and 3 in network_ids:
            warnings.warn(
                "Asked to return data from 2-D convolutional networks but no image data was given. Ignoring invalid requests.",
                Warning)
            while 3 in network_ids:
                try:
                    network_ids.remove(3)
                except ValueError:
                    pass
        assert len(network_ids) is not 0, "No valid network id's specified for the input data types."

        predictions = np.empty((batch_size, len(network_ids), 2))
        coordinates = []
        if targets is not None:
            errors = np.empty((batch_size, len(network_ids), 2))

        if verbose:
            print "Solving..."
        if timing:
            start = time.time()
            if verbosity >= 2:
                times = [start]
        if batch_process:
            for id in range(len(network_ids)):
                if network_ids[id] in [0, 1, 2]:
                    batch_prediction = self.nets[network_ids[id]](input1D)
                    predictions[:, id, :] = [self.to_output(sample) for sample in batch_prediction]
                elif network_ids[id] in [3, 4, 5]:
                    batch_prediction = self.nets[network_ids[id]](input2D)
                    predictions[:, id, :] = [self.to_output(sample) for sample in batch_prediction]
            if targets is not None:
                for batch in range(batch_size):
                    errors[batch, id, :] = self.get_error(prediction[batch], targets[batch], relative, epsilon)
        else:
            for batch in range(batch_size):
                for id in range(len(network_ids)):
                    if network_ids[id] in [0, 1, 2]:
                        prediction = self.to_output(self.nets[network_ids[id]](input1D)[0])
                        predictions[batch, id, :] = prediction
                    elif network_ids[id] in [3]:
                        prediction = self.to_output(self.nets[network_ids[id]](input2D)[0])
                        predictions[batch, id, :] = prediction
                    if timing:
                        if verbose and verbosity >= 2:
                            times.append(time.time())
                            print "Sample {} solution time: {:.3f}s".format(batch, times[-1] - times[-2])
                    if targets is not None:
                        errors[batch, id, :] = self.get_error(prediction, targets[batch], relative, epsilon)
        if feedback:
            try:
                self.feedbackdir = os.path.join(self.netdir, "../feedback/" + str(self.logtime))
                os.makedirs(self.feedbackdir)
            except:
                raise
            for i in range(batch_size):
                scale = np.mean(predictions[i, :, 0])
                print "scale predictions"
                print predictions[i, :, 0]
                # scales = np.linspace(max(sys.float_info.epsilon, scale - .2), scale + .2, 10)
                print "Angle predictions"
                print predictions[i, :, 1]
                angle = np.mean(predictions[i, :, 1])
                # angles = np.linspace(angle
                #                      - 0.2 * np.pi, angle + 0.2 * np.pi, 10)
                # accumulator = general_hough_closure(self.ref_image, angles=angles, scales=scales, show_progress=False)
                # acc_array, scales, angles = accumulator(input2D[i])
                # y, x, s, a = np.unravel_index(acc_array.argmax(), acc_array.shape)
                # dy, dx = y, x
                # coordinates.append((dy, dx, scales[s], angles[a]))
                # coordinates.append((y, x, scales[s], angles[a]))
                # print "Coordinates"
                # print coordinates
                # filename = "feedback_" + str(i) + "_" + str(dy) + "_" + str(dx) + "_" + str(
                #     angles[a] * 180. / np.pi) + "_" + str(scales[s]) + ".png"
                filename = "feedback_" + str(i) + "_" + str(angle) + "_" + str(scale) + ".png"

                overlay = np.copy(self.ref_image)
                # overlay = zoom(overlay, scales[s])
                overlay = zoom(overlay, scale)
                # overlay = rotate(overlay, angles[a], reshape=False)
                overlay = rotate(overlay, angle, reshape=False)
                # overlay = shift(overlay, (dy, dx))
                # y_crop = slice(max(0, overlay.shape[0]/2-self._input2D_shape[0]/2), max(overlay.shape[0]/2+self._input2D_shape[0]/2, overlay.shape[0]))
                # x_crop = slice(max(0, overlay.shape[1]/2-self._input2D_shape[1]/2), max(overlay.shape[1]/2+self._input2D_shape[1]/2, overlay.shape[1]))
                print "Overlay shape"
                print overlay.shape
                # overlay = overlay[y_crop, x_crop]
                print overlay.shape
                plt.imshow(overlay, alpha=0.5)
                original = np.copy(input2D[i])
                original = rgb_to_grey(original)
                plt.imshow(original, alpha=0.5)
                # print coordinates[-1]
                # print coordinates[-1][0]
                # print coordinates[-1][1]
                # print coordinates[-1][2]
                # print coordinates[-1][3]
                plt.savefig(os.path.join(self.feedbackdir, filename))
                plt.clf()
                f, axarr = plt.subplots(2)
                axarr[0].imshow(original)
                axarr[1].imshow(overlay)
                plt.savefig(os.path.join(self.feedbackdir, "subplots" + filename))
                plt.clf()
        if verbose:
            print "Done"

        if timing and verbose:
            stop = time.time() - start
            if batch_process:
                print "--" * 50
                print "Total Batch Time: {:.3f}s".format(stop)
                print "##" * 50
                print
            else:
                print "--" * 50
                print "Total Time: {:.3f}s".format(stop)
                if verbosity >= 2:
                    times = np.diff(times)
                    print "Mean time: {:.3f}s".format(np.mean(times))
                    print "Median time: {:.3f}s".format(np.median(times))
                    print "Max time: {:.3f}s".format(np.max(times))
                    print "Min time: {:.3f}s".format(np.min(times))
                    print "Standard deviation: {:.3f}s".format(np.std(times))
                    print "##" * 50
                    print

        if targets is None:
            return predictions, networks, coordinates
        else:
            return predictions, networks, coordinates, errors


def test(pickled):
    if not pickled:
        # Read Images
        dir = os.path.dirname(__file__)
        ref_im_file = os.path.join(dir, "../images/TEST/reference.jpg")

        grey_ref = sp_imread(ref_im_file, flatten=True)
        color_ref = sp_imread(ref_im_file)

        print "Segmenting Ref"
        segmented_ref = segment(grey_ref, color_ref, ref_im_file, 50, itermagnification=2, debug=False,
                                scale=grey_scale, minsegmentarea=0.1, datadir="./")

        ref_leaflets = []
        ref_names = []
        ref_scales = []
        ref_contours = []

        for i in range(segmented_ref[3]):
            leaf, scale, contour = iso_leaf(segmented_ref[0], i + 1, ref_image=color_ref)
            ref_leaflets.append(leaf)
            ref_scales.append(scale)
            ref_contours.append(contour)
            ref_names.append("leaflet %s" % i)

        # False leaflet
        ref_leaflets.pop(6)
        ref_scales.pop(6)
        ref_contours.pop(6)
        ref_names.pop(6)
        print "Done"
        print
        data = TrainingData(ref_leaflets, ref_contours, ref_scales, names=ref_names, sigma=0.075 * 64)

    if not pickled:
        print 'Generating data...'
        inputs, inputs2D, targets, weights = data.generatedata(ninputs=100)
        print "Data generated"
        print
        print "Pickling data..."
        cPickle.dump(data, open("leafdump.pkl", "wb"))
        cPickle.dump(inputs, open("inputdump.pkl", "wb"))
        cPickle.dump(inputs2D, open("input2Ddump.pkl", "wb"))
        cPickle.dump(targets, open("targetdump.pkl", "wb"))
        print 'Done'
        print
    else:
        print "Loading pickled data..."
        data = cPickle.load(open("leafdump.pkl", "rb"))
        inputs = cPickle.load(open("inputdump.pkl", "rb"))
        inputs2D = cPickle.load(open("input2Ddump.pkl", "rb"))
        targets = cPickle.load(open("targetdump.pkl", "rb"))
        print "Done"
        print

    inputs = np.asarray(inputs)
    targets = np.asarray(targets)

    pickled = False
    if pickled:
        print "Loading pickled network..."
        LeafNet = cPickle.load(open("netdump.pkl", "rb"))
        print "Net loaded"
        print
    else:
        print 'Building network...'
        LeafNet = LeafNetwork(inputs, inputs2D, targets, data.leaves[0].color_image)
        print 'Network constructed'
        print
        print "Pickling network..."
        cPickle.dump(LeafNet, open("netdump.pkl", "wb"))
        print "Done"
        pickled = True


if __name__ == '__main__':
    pickled = True
    # pickled = False
    # if pickled:
    #     print "Loading pickled objects..."
    #     data = cPickle.load(open("leafdump.pkl", "rb"))
    #     inputs = cPickle.load(open("inputdump.pkl", "rb"))
    #     targets = cPickle.load(open("targetdump.pkl", "rb"))
    # LeafNet = cPickle.load(open("netdump.pkl", "rb"))
    # print "Done"
    pickled = test(pickled)
