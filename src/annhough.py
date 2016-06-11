ninputs = 100

# Standard Imports
import os
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

def get_scale(leafimage, minticks, scaleunits):
    """

    :param leafimage:
    :param minticks:
    :param scaleunits:
    :return:
    """
    print "get_scale"
    n, m = leafimage.shape

    scale, found, edge = metricscale(leafimage, minticks, scaleunits)
#        print "width,height (cm) = ", n * scale, m * scale, "scale = ", scale
    if not found:   # try to find a scale after histogram equalization
        scale, found, edge = metricscale(leafimage, minticks, scaleunits, True, False)
#            print "width,height with histogram equalization (cm) = ", n * scale, m * scale, "scale = ", scale

    if not found:   # try to find a scale after Gaussian blur
        scale, found, edge = metricscale(leafimage, minticks, scaleunits, False, True)
#                print "width,height with histogram equalization (cm) = ", n * scale, m * scale, "scale = ", scale
    return scale

def iso_leaf(segmented_image, leaf_num, square_length=64, ref_image=None, rescale=True):
    """
    Takes a segmented image containing a leaf and returns a cropped (and maybe scaled) image containing just leaf
    specified by leaf_num by isolating the pixels with value leaf_num.
    :param segmented_image: ndarray
                        The segmented image containing the leaf to be isolated.
    :param leaf_num: int
                        The number specifying which leaf to isolate.
    :param square_length: int, optional
                        The dimension which the final output image should be. Default is 64.
    :param ref_image:   ndarray, optional
                        If specified, isolates the leaf in this image as well. Useful if color information must be
                        preserved.
    :param rescale:     Bool, optional
                        Whether to allow rescaling of the isolated leaf images.
    :return: petiole, scale, contour
            petiole: ndarray, the image of the isolated leaf
            scale: float, the scale used to make the image fit into the dimension specified by square_length
            contour: contour, a contour instance of the leaf
    """
    scale = 1.
    rows, cols = np.where(segmented_image == leaf_num)
    left = np.min(cols)
    top = np.min(rows)
    dimspan = max(np.ptp(zip(rows, cols), 0))

    if ref_image is not None:
        ref_image = ref_image.astype(float)
        petiole = np.zeros(ref_image.shape, float)
        for p in zip(rows, cols):
            petiole[p] = ref_image[p]
        petiole = petiole[top:(top+dimspan+1), left:(left+dimspan+1)]
    else:
        petiole = np.zeros(segmented_image.shape[:2], bool)
        i = 0
        for pixel in zip(rows, cols):
            petiole[pixel] = True
            i+=1
        petiole = petiole[top:(top + dimspan + 1), left:(left + dimspan + 1)]

    contour = np.zeros(segmented_image.shape[:2], bool)
    for c in zip(rows, cols):
        contour[c] = True
    contour = contour[top:(top + dimspan + 1), left:(left + dimspan + 1)]
    if rescale:
        scale = float(square_length-2)/float(max(petiole.shape[:2]))
        petiole = imresize(petiole, scale, interp='bicubic')
        contour = imresize(contour, scale, interp='bicubic').astype(bool)
        contour = np.pad(contour, 1, mode='constant', constant_values=False)
    try:
        contour = parametrize(contour)
    except TypeError:
        print "No leaf found in image"
        raise

    if ref_image is not None:
        padded = np.zeros((square_length, square_length, 3), float)
        shape = [s+1 for s in petiole.shape[:2]]
        if shape[0] < square_length+1 and shape[1] < square_length+1:
            padded[1:shape[0], 1:shape[1]] = petiole[:, :]
        elif shape[1] < square_length+1:
            padded[1:, 1:shape[1]] = petiole[:63, :]
        else:
            padded[1:, 1:] = petiole[:square_length-1, :square_length-1]
        petiole = padded
    else:
        padded = np.zeros((square_length, square_length), bool)
        shape = [s+1 for s in petiole.shape[:2]]
        try:
            if shape[0] < square_length+1 and shape[1] < square_length+1:
                padded[1:shape[0], 1:shape[1]] = petiole[:, :]
            elif shape[1] < square_length+1:
                padded[1:, 1:shape[1]] = petiole[:square_length-1, :]
            elif shape[0] < square_length+1:
                padded[1:shape[0], 1:] = petiole[:, :square_length-1]
            else:
                padded[1:, 1:] = petiole[:square_length-1, :square_length-1]
        except TypeError:
            print "Petiole shape: ", shape
            raise
        petiole = padded
    return petiole, scale, contour

def gradient_orientation(image):
    """
    Calculate the gradient orientation for edge point in the image.
    :param image: ndarray
                Image for which the gradient orientation should be calculated.
    :return: gradient
                ndarray of gradient directions at each point in the image.
    """
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')

    gradient = np.arctan2(dy,dx) * 180 / np.pi

    return gradient

def parametrize(image):
    """
    This function parameterizes an object in the image using a boundary following algorithm.
    :param image: ndarray
                    image containing the object to be parameterized
    :return: contour
                    a list of tuples of the pixel locations of the object boundary
    """
    contour = []
    # Find the upper-left-most edge pixel for boundary following
    image = image.astype(bool)
    for (j,k), value in np.ndenumerate(image):
        if value:
            b0 = (j, k)
            c0 = (j, k-1)
            contour.append(b0)
            break

    # Find the next contour point b1 to use as stop condition
    try:
        b1, c1 = check8(image, b0, c0)
    except TypeError:
        print b0, c0
        plt.imshow(image)
        plt.show()
        print type(image)
        print image.shape
        print image[0, 0]
        raise
    contour.append(b1)
    b, c = b1, c1
    stop = image.shape[0]*image.shape[1]
    # Follow the boundary until we return to b0 -> b1
    s = 0
    while True:
        bit, cit = check8(image, b, c)
        if b == b0 and bit == b1 or s == stop:
            break
        b, c = bit, cit
        s+=1
        contour.append(b)
    return contour

def check8(image, fgpixel, bgpixel):
    """
     Checks the 8 neighbors of fgpixel for the first true value in a clockwise orientation starting from bgpixel
    :param image: ndarray
                image in which to search
    :param fgpixel: tuple
                foreground pixel location
    :param bgpixel:
                background pixel location
    :return: tuple, tuple
                tuples denoting the first true pixel location found, and the last false pixel before the true pixel was
                found.
    """

    orientation = convert((bgpixel[0]-fgpixel[0], bgpixel[1]-fgpixel[1]))
    movements = [int(orientation + i + 1)%8 for i in range(8)]
    for m in movements:
        try:
            if image[fgpixel[0] + convert(m)[0], fgpixel[1] + convert(m)[1]]:
                return (fgpixel[0] + convert(m)[0], fgpixel[1] + convert(m)[1]), (fgpixel[0] + convert(m-1)[0], fgpixel[1] + convert(m-1)[1])
        except IndexError:
            pass

def convert(Input):
    """
    Helper function to convert between chain-code and relative position representations.
    :param Input: int or tuple
                int: convert Input from chain-code to relative position
                tuple: covert Input from relative position to chain-code

    :return: tuple or int
                tuple: the relative position representation of Input
                int: the chain-code representation of Input
    """
    PositionToCode = {(0, -1):0, (-1, -1):1, (-1, 0):2, (-1, 1):3, (0, 1):4, (1, 1):5, (1, 0):6, (1, -1):7}
    CodeToPosition = {0:(0, -1), 1:(-1, -1), 2:(-1, 0), 3:(-1, 1), 4:(0, 1), 5:(1, 1), 6:(1, 0), 7:(1, -1)}
    if type(Input) is tuple:
        return PositionToCode[Input]
    if type(Input) is int or float:
        return CodeToPosition[int(Input)%8]

class TrainingData:
    def __init__(self, leaves, contours, scales, sigma=1., img_from_contour=False, names=None):
        print 'Training Data'
        self.sigma = sigma
        if names is None:
            names = self.namegen(names)
        self.leaves             = [Leaf(leaves[_],
                                        contour=contours[_],
                                        scale=scales[_],
                                        sigma=self.sigma,
                                        img_from_contour=img_from_contour,
                                        name=names[_],
                                        rescale=True) for _ in range(len(leaves))]
        self.contours           = [leaf.contour for leaf in self.leaves]
        self.trainingdata       = list()
        self.trainingdata2D     = list()
        self.trainingtargets    = list()
        self.trainingweights    = list()
        self.update()
        print "Initial Leaves Added"
        self.eaten_leaves = list()
    def namegen(self):
        i = 0
        print 'namegen'
        while True:
            yield "training leaf %r " %i
            i += 1
            if i == len(leaves):
                break
    @property
    def __len__(self):
        self.len=len(self.leaves)
        return self.len
    def update(self, sigma=None, visualize=False):
        """Bookkeeping method to make sure all data is consistent"""
        if sigma is not None:
            self.sigma = sigma
        self.smoothcontour(self.sigma)
        self.orient(visualize=visualize)
        self.findcentroid()
        self.getangles(sigma = self.sigma)
        self.curvature(self.sigma)
    def curvature(self, sigma=None, indeces=None):
        '''Computes the curvature at each point along the edge for each leaf specified by indeces'''
        if sigma is not None:
            self.sigma = sigma
        curves = list()
        if indeces is None:
            for leaf in self.leaves:
                leaf.curvature(sigma=self.sigma)
                curves.append(leaf.curvatures)
        else:
            indeces = list(indeces)
            for index in indeces:
                self.leaves[index].curvature(sigma=self.sigma)
                curves.append(leaves[index].curvatures)
        return curves
    def smoothcontour(self, sigma=1., indeces=None):
        """Smooth all the contours by convolving a Gaussian kernel with a width given by sigma. If indeces is specified
        then only those leaves' contours are updated"""
        self.sigma = sigma
        smoothedc = list()

        if indeces is None:
            for leaf in self.leaves:
                leaf.smooth(self.sigma)
                smoothedc.append(leaf.smooth_contour)
        else:
            indeces = list(indeces)
            for index in indeces:
                self.leaves[index].smooth(self.sigma)
                smoothedc.append(self.leaves[index].smooth_contour)
        return smoothedc
    def findcentroid(self, indeces=None):
        """Find the centroids of each of the leaves. If indeces is specified then only find those centroids"""
        centroids = list()
        if indeces is None:
            for leaf in self.leaves:
                leaf.findcentroid()
                centroids.append(leaf.centroid)
        else:
            indeces = list(indeces)
            for index in indeces:
                self.leaves[index].findcentroid()
                centroids.append(self.leaves[index].centroid)
        return centroids
    def getangles(self, indeces=None, **kwargs):
        """Calculates the angle from the centroid to each edge pixel. Returns the list of angles to pixel locations for
        the leaf at index. If no index is specified, returns a list containing the lists of angles for each leaf"""
        smoothedc = list()
        if indeces is None:
            for leaf in self.leaves:
                leaf.getangles(**kwargs)
                smoothedc.append(leaf.getangles())
        else:
            indeces = list(indeces)
            for index in indeces:
                self.leaves[index].getangles(**kwargs)
                smoothedc.append(self.leaves[index].getangles())
        return smoothedc
    def orient(self, indeces=None, **kwargs):
        """Vertically orient the leaves given by indeces. If indeces is not specified then all leaves are oriented"""
        if indeces is None:
            for leaf in self.leaves:
                leaf.orient(**kwargs)
        else:
            indeces = list(indeces)
            for index in indeces:
                self.leaves[index].orient(**kwargs)
    def generatedata(self, ninputs=100, base_size=20, verbose=False, **kwargs):
        """Generate data to be used by the neural networks.
                base_size: number of basic types of eaten leaves to be further scaled and rotated."""
        self.trainingdata = []
        self.trainingdata2D = []
        self.trainingtargets = []
        self.trainingweights = []
        angles = np.linspace(0., 2.*np.pi*(1.-1./float(ninputs)), ninputs)
        for leaf in self.leaves:
            leaf.generatewalks(20)
            data, targets, weights = self.randomdata(leaf, **kwargs)
            print "Data length:"
            print len(data)
            for d in data:
                cdata = list()
                d.curvature(self.sigma)
                print "Image Shape: ", d.color_image.shape
                for a in angles:
                    try:
                        c = d.curvatures[d.extractpoint(a)]
                        if np.isnan(c):
                            print "C IS STILL A NAN"
                            print
                            c = 1E9
                        cdata.append(c)
                    except ValueError:
                        print "Curvature Data ValueError"
                        print "Warning: Curvature not defined for angle %r in leaf %r" %(180.*a/np.pi, d.__name__)
                        print "Substituting with %r instead" % leaf.__name__
                        cdata.append(0)
                        index = data.index(d)
                        targets[index] = (1, 0)
                        weights[index] = 1
                        cdata = [leaf.curvatures[leaf.extractpoint(a)] for a in angles]
                        break
                self.trainingdata.append(cdata)
                # Transpose the and reshape data to be accepted by lasagne according to (channels, height, width)
                dimage = d.color_image.transpose(2, 0, 1)
                dimage = dimage.reshape(1, dimage.shape[0], dimage.shape[1], dimage.shape[2])
                print "New Shape: ", dimage.shape
                print
                self.trainingdata2D.append(dimage)
            self.trainingtargets.extend(targets)
            self.trainingweights.extend(weights)
        self.trainingdata2D = np.concatenate(self.trainingdata2D, axis=0)
        print self.trainingdata2D.shape
        return self.trainingdata, self.trainingdata2D, self.trainingtargets, self.trainingweights
    def randomdata(self, leaf, data_size=100, square_length=64, **kwargs):
        """Creates a list of randomly scaled and rotated partially eaten leaves"""
        print 'Random Data'
        data = list()
        targets = list()
        weights = list()
        leaf_list = list()
        weight = list()
        base_size = kwargs.get('base_size', 20)
        for i in range(base_size):
            d, w = leaf.randomwalk(**kwargs)
            leaf_list.append(d)
            weight.append(w)
        size = 0
        while size < data_size:
            scale = 1.5*np.random.rand()+0.5
            angle = 2.*np.pi*np.random.rand()-np.pi
            index = np.random.randint(0, len(leaf_list))
            leaf = leaf_list[index]

            if not leaf.image.any():
                print "None, ever"
                plt.imshow(leaf.image)
                plt.show()
                plt.imshow(leaf.image.astype(bool))
                plt.show()
                continue
            new_leaf = tf.rotate(leaf.image, angle*180./np.pi, resize=True)
            if not new_leaf.any():
                print "None after TF"
                plt.imshow(new_leaf)
                plt.show()
                continue
            new_leaf = imresize(new_leaf, scale, interp='bicubic')
            if not new_leaf.any():
                print "None after RZ"
                plt.imshow(new_leaf)
                plt.show()
                continue

            leaf_c = tf.rotate(leaf.cimage, angle*180./np.pi, resize=True)
            leaf_c = imresize(leaf_c, scale, interp='bicubic')
            try:
                ubounds = np.amax(parametrize(leaf_c), 0)
                lbounds = np.amin(parametrize(leaf_c), 0)
            except TypeError:
                print "Leaf Images"
                print "Leaf_C"
                plt.imshow(leaf_c)
                plt.show()
                print "NewLeaf"
                plt.imshow(new_leaf)
                plt.show()
                print
                continue

            leaf_c = parametrize(leaf_c)
            if not new_leaf.any():
                print leaf.__name__
                print np.amax(leaf_c, 0)
                print angle*180./np.pi
                print scale
                img = np.zeros(np.amax(leaf_c, 0)+1, bool)
                for c in leaf_c:
                    img[c] = True
                plt.imshow(img)
                plt.show()
                plt.imshow(new_leaf)
                plt.show()
                print
                continue

            # Crop the images to 64x64
            shape = new_leaf.shape
            padded = np.zeros((square_length, square_length, 3), dtype=np.uint8)
            try:
                for i in range(3):
                    if shape[0] <= square_length and shape[1] <= square_length:
                        padded[1:shape[0], 1:shape[1], i] = new_leaf[:-1, :-1]
                    elif shape[1] < square_length + 1:
                        padded[1:, 1:shape[1], i] = new_leaf[:square_length - 1, :]
                    elif shape[0] < square_length + 1:
                        padded[1:shape[0], 1:, i] = new_leaf[:, :square_length - 1]
                    else:
                        padded[1:, 1:, i] = new_leaf[:square_length - 1, :square_length - 1]
            except TypeError:
                print "Image shape: ", shape
                raise
            new_leaf = padded
            try:
                new_leaf = Leaf(new_leaf, contour=leaf_c, scale=scale, sigma=leaf.sigma, orient=False, rescale=False, name='eaten leaf %r of %r' %(size, leaf.__name__))
            except TypeError:
                print "TypeError"
                print leaf.__name__
                try:
                    print "New Leaf.image"
                    print new_leaf
                    print type(new_leaf)
                    plt.imshow(new_leaf.image)
                    plt.show()
                    raise
                except AttributeError:
                    print "New Leaf Displayed"
                    plt.imshow(new_leaf)
                    plt.show()
                    raise
                continue
            data.append(new_leaf)
            targets.append((scale, angle))
            weights.append(weight[index])
            size += 1
        return data, targets, weights

class LeafNetwork(object):
    def split_data(self, train_prop=0.75, verbose=False, **kwargs):
        """Splits the data into train_prop% training and 1-train_prop% validation data"""
        if verbose:
            print "Splitting Data..."
        inx     = int(train_prop*len(self._data1D))
        self.X_train, self.y_train  = self._data1D[:inx], self._targets[:inx]
        self.X_train2D = self._data2D[:inx]
        self.X_val, self.y_val      = self._data1D[inx:], self._targets[inx:]
        self.X_val2D = self._data2D[inx:]
        if verbose:
            print "Done"
            print
    def __init__(self, data1D, data2D, targets, train_prop=0.75, num_epochs=1000, stop_err=0.01,
                 d_stable=1e-6, n_stable=10, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, nonlinearity='tanh', n_1Dfilters=2, n_2Dfilters=6, n_conv=2, n_dense=3,
                 freeze_autoencoder=False, verbose=True, plotting=True, verbosity=3, **kwargs):
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
            :param n_1Dfilters:    int, optional
                            Number of learnable filters each 1D convolutional layer should have. Default is 2.
            :param n_2Dfilters:    int, optional
                            Number of learnable filters each 2D convolutional layer should have. Default is 6.
            :param n_conv:         int, optional
                            Number of convolutional layers the networks should have. Must be at least 2 in order to
                            avoid excessive training times and accurate results. Default is 2
            :param n_dense:        int, optional
                            Number of dense, fully connected layers the networks should have. Must be at least 2 in
                            order to ensure accuracy. Default is 3.
            :param freeze_autoencoder: Bool, optional
                            Whether or not to freeze the autoencoding layers after pretraining. Default is True.
            :param verbose: Bool, optional
                            Whether or not to print messages to the console. Default is set to True.
            :param plotting: Bool, optional
                            Whether or not to show plots of the training and validation errors for each network after
                            pretraining and training. Default is True.
            :param verbosity: int, optional
                            Set the verbosity level.
                            0: low verbosity
                            3: maximum verbosity
        """
        self._nonlindict    = {'tanh': nonlinearities.tanh, 'sigmoid': nonlinearities.sigmoid,
                               'softmax': nonlinearities.softmax, 'ReLU': nonlinearities.rectify,
                               'linear': nonlinearities.linear, 'exponential': nonlinearities.elu,
                               'softplus': nonlinearities.softplus, 'identity': nonlinearities.identity}

        assert n_conv > 1, "Too few convolutional layers. Must have at least 2.\nn_conv: %r" % (n_conv)
        assert n_dense > 2, "Too few dense layers. Must have at least 2.\nn_dense: %r" % (n_dense)
        assert len(data1D.shape) == 2, "1D data shape error: training data has wrong number of dimensions.\nExpected 2 with format (# samples, points/sample) and instead got %r" % (data1D.shape)
        assert len(data2D.shape) == 4, "2D data shape error: training data has wrong number of dimensions.\nExpected 4 with format (# samples, channels/sample, height, width) and instead got %r" % (data2D.shape)

        self._input1D_size = data1D.shape[-1]
        self._input2D_shape = data2D.shape[1:]

        for i in range(data1D.shape[0]):
            targets[i, 0]   = np.log2(targets[i, 0])
            targets[i, 1]   = targets[i, 1]/(2.*np.pi)-.5
        new_data = np.zeros((data1D.shape[0], 1, data1D.shape[1]))
        for i in range(data1D.shape[0]):
            new_data[i, 0, :]   = data1D[i, :]

        self._data1D    = new_data
        self._data2D    = data2D
        self._targets   = targets
        self.split_data(train_prop)
        # self.create_layers(n_1Dfilters, n_2Dfilters, n_conv, n_dense, nonlinearity, freeze_autoencoder, verbose=verbose, verbosity=verbosity, **kwargs)
        self.create_layers(8, 24, n_conv, n_dense, nonlinearity, freeze_autoencoder, verbose=verbose, verbosity=verbosity, **kwargs)
        self.train_network(num_epochs, stop_err, d_stable, n_stable, learning_rate, beta_1, beta_2, epsilon, verbose=verbose, plotting=plotting, verbosity=verbosity, **kwargs)



    def create_layers(self, n_1Dfilters=2, n_2Dfilters=6, n_conv=2, n_dense=3, nonlinearity='tanh', freeze_autoencoder=False, **kwargs):
        """
        Builds 1- and 2-D Convolutional Networks using Theano and Lasagne
        """
        verbose = kwargs.get("verbose", False)
        verbosity = kwargs.get("verbosity", 2)

        self.input_var       = T.tensor3('inputs')
        self.input_var2D     = T.tensor4('2D inputs')
        self.AE_target_var   = T.tensor3('AE inputs')
        self.AE_target_var2D = T.tensor4('AE 2D targets')
        self.target_var      = T.matrix('targets')

        self._nonlinearity  = self._nonlindict[nonlinearity]
        self.nonlinearity   = nonlinearity
        self._n_conv        = n_conv
        self._n_dense       = n_dense
        self._n_1Dfilters   = n_1Dfilters
        # self._n_1Dfilters   = 1
        self._n_2Dfilters   = n_2Dfilters
        self._frozen        = freeze_autoencoder

        pool_size   = kwargs.get('pool_size', 2)
        dropout     = kwargs.get('dropout', 0.5)
        filter_size = kwargs.get('filter_size', 3)

        if verbose: print "Layering Networks..."
        self.AE2DLayers     = []
        self.AELayers       = []
        self.ConvLayers     = []
        self.Conv2DLayers   = []

        # Input Layer
        # 1D
        self.DConvLayers = layers.InputLayer((None, 1, self._data1D.shape[-1]), input_var=self.input_var)
        self.DenseLayers = layers.InputLayer((None, 1, self._data1D.shape[-1]), input_var=self.input_var)
        self.AELayers.append(layers.InputLayer((None, 1, self._data1D.shape[-1]), input_var=self.input_var))
        self.ConvLayers.append(layers.InputLayer((None, 1, self._data1D.shape[-1]), input_var=self.input_var))
        self.DConvLayers = layers.batch_norm(layers.Conv1DLayer(self.DConvLayers, num_filters=1, filter_size=3, nonlinearity=self._nonlinearity)) # no nonL, 1 filt
        # 2D
        self.AE2DLayers.append(layers.InputLayer((None, self._data2D.shape[1], self._data2D.shape[2], self._data2D.shape[3]), input_var=self.input_var2D))
        self.Conv2DLayers.append(layers.InputLayer((None, 3, 64, 64), input_var=self.input_var2D))

        # Batch Normalization
        # 1D
        self.AELayers.append(layers.BatchNormLayer(self.AELayers[-1]))
        self.DenseLayers = layers.BatchNormLayer(self.DenseLayers)
        self.ConvLayers.append(layers.BatchNormLayer(self.ConvLayers[-1], alpha=self.AELayers[-1].alpha, beta=self.AELayers[-1].beta, gamma=self.AELayers[-1].gamma, mean=self.AELayers[-1].mean, inv_std=self.AELayers[-1].inv_std))

        # 2D
        self.AE2DLayers.append(layers.BatchNormLayer(self.AE2DLayers[-1]))
        self.Conv2DLayers.append(layers.BatchNormLayer(self.Conv2DLayers[-1], alpha=self.AE2DLayers[-1].alpha, beta=self.AE2DLayers[-1].beta, gamma=self.AE2DLayers[-1].gamma, mean=self.AE2DLayers[-1].mean, inv_std=self.AE2DLayers[-1].inv_std))
        ###########################################################################################################

        # units = [] # store the output shapes for creating the analagouce dense network
        for c in range(n_conv-1):
            # Convolutional and Pooling Layers 1
            # 1D
            self.AELayers.append(layers.Conv1DLayer(self.AELayers[-1], num_filters=n_1Dfilters, filter_size=filter_size, nonlinearity=self._nonlinearity))
            self.ConvLayers.append(layers.Conv1DLayer(self.ConvLayers[-1], num_filters=n_1Dfilters, filter_size=filter_size, W=self.AELayers[-1].W, b=self.AELayers[-1].b, nonlinearity=self._nonlinearity))
            print "Output shapes after convolution"
            print layers.get_output_shape(self.ConvLayers[-1])
            self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout), num_units=layers.get_output_shape(self.ConvLayers[-1])[-1], nonlinearity=self._nonlinearity)
            print layers.get_output_shape(self.DenseLayers)
            if c is not 0:
                # DConvLayers already has a convolutional layer from the batch normalization step, so skip on the first iteration.
                self.DConvLayers = layers.Conv1DLayer(self.DConvLayers, num_filters=1, filter_size=filter_size, nonlinearity=self._nonlinearity)

            print "Output shapes after pooling"
            self.AELayers.append(layers.MaxPool1DLayer(self.AELayers[-1], pool_size=pool_size))
            self.ConvLayers.append(layers.MaxPool1DLayer(self.ConvLayers[-1], pool_size=pool_size))
            print layers.get_output_shape(self.ConvLayers[-1])
            self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout), num_units=layers.get_output_shape(self.ConvLayers[-1])[-1], nonlinearity=self._nonlinearity)
            print layers.get_output_shape(self.DenseLayers)
            # units.append[layers.get_output_shape(self.ConvLayers[-1])]
            self.DConvLayers = layers.MaxPool1DLayer(self.DConvLayers, pool_size=pool_size)

            # 2D
            self.AE2DLayers.append(layers.Conv2DLayer(self.AE2DLayers[-1], num_filters=n_2Dfilters, filter_size=filter_size, nonlinearity=self._nonlinearity))
            self.Conv2DLayers.append(layers.Conv2DLayer(self.Conv2DLayers[-1], num_filters=n_2Dfilters, filter_size=filter_size, W=self.AE2DLayers[-1].W, b=self.AE2DLayers[-1].b, nonlinearity=self._nonlinearity)) # no nonL, 3 filt

            self.AE2DLayers.append(layers.MaxPool2DLayer(self.AE2DLayers[-1], pool_size=pool_size))
            self.Conv2DLayers.append(layers.MaxPool2DLayer(self.Conv2DLayers[-1], pool_size=pool_size))

            if freeze_autoencoder:
                self.ConvLayers[-1].params[self.ConvLayers[-1].W].remove("trainable")
                self.ConvLayers[-1].params[self.ConvLayers[-1].b].remove("trainable")
                self.Conv2DLayers[-1].params[self.Conv2DLayers[-1].W].remove("trainable")
                self.Conv2DLayers[-1].params[self.Conv2DLayers[-1].b].remove("trainable")
            ###########################################################################################################

            # Convolutional and Pooling Layers output
            #  1D
            self.AELayers.append(layers.Conv1DLayer(self.AELayers[-1], num_filters=n_1Dfilters, filter_size=filter_size, nonlinearity=self._nonlinearity))
            self.ConvLayers.append(layers.Conv1DLayer(self.ConvLayers[-1], num_filters=n_1Dfilters, filter_size=filter_size, W=self.AELayers[-1].W, b=self.AELayers[-1].b, nonlinearity=self._nonlinearity))
            print "Output shapes after convolution"
            print layers.get_output_shape(self.ConvLayers[-1])
            self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout), num_units=layers.get_output_shape(self.ConvLayers[-1])[-1], nonlinearity=self._nonlinearity)
            print layers.get_output_shape(self.DenseLayers)
            self.DConvLayers = layers.Conv1DLayer(self.DConvLayers, num_filters=1, filter_size=filter_size, nonlinearity=self._nonlinearity)

            self.AELayers.append(layers.MaxPool1DLayer(self.AELayers[-1], pool_size=pool_size))
            self.ConvLayers.append(layers.MaxPool1DLayer(self.ConvLayers[-1], pool_size=pool_size))
            print "Output shapes after pooling"
            print layers.get_output_shape(self.ConvLayers[-1])
            self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout), num_units=layers.get_output_shape(self.ConvLayers[-1])[-1], nonlinearity=self._nonlinearity)
            print layers.get_output_shape(self.DenseLayers)
            self.DConvLayers = layers.MaxPool1DLayer(self.DConvLayers, pool_size=pool_size)

            # 2D
            self.AE2DLayers.append(layers.Conv2DLayer(self.AE2DLayers[-1], num_filters=n_2Dfilters, filter_size=filter_size, nonlinearity=self._nonlinearity))
            self.Conv2DLayers.append(layers.Conv2DLayer(self.Conv2DLayers[-1], num_filters=n_2Dfilters, filter_size=filter_size, W=self.AE2DLayers[-1].W, b=self.AE2DLayers[-1].b, nonlinearity=self._nonlinearity))

            self.AE2DLayers.append(layers.MaxPool2DLayer(self.AE2DLayers[-1], pool_size=pool_size))
            self.Conv2DLayers.append(layers.MaxPool2DLayer(self.Conv2DLayers[-1], pool_size=pool_size))

            # Add Decoding Layers
            down = len(self.AELayers)
            for i in range(down-1):
                self.AELayers.append(layers.InverseLayer(self.AELayers[-1], self.AELayers[down - 1 - i]))
                self.AE2DLayers.append(layers.InverseLayer(self.AE2DLayers[-1], self.AE2DLayers[down - 1 - i]))

            if freeze_autoencoder:
                self.ConvLayers[-1].params[self.ConvLayers[-1].W].remove("trainable")
                self.ConvLayers[-1].params[self.ConvLayers[-1].b].remove("trainable")
                self.Conv2DLayers[-1].params[self.Conv2DLayers[-1].W].remove("trainable")
                self.Conv2DLayers[-1].params[self.Conv2DLayers[-1].b].remove("trainable")
            ###########################################################################################################

            # Dense Layer 1
            self.DConvLayers = layers.DenseLayer(layers.dropout(self.DConvLayers, p=dropout), num_units=2*layers.get_output_shape(self.DConvLayers)[-1], nonlinearity=self._nonlinearity)
            self.Conv1DScaleNetLayers = layers.DenseLayer(layers.dropout(self.ConvLayers[-1], p=dropout), num_units=2*layers.get_output_shape(self.ConvLayers[-1])[-1], nonlinearity=self._nonlinearity)
            self.Conv1DAngleNetLayers = layers.DenseLayer(layers.dropout(self.ConvLayers[-1], p=dropout), num_units=2*layers.get_output_shape(self.ConvLayers[-1])[-1], nonlinearity=self._nonlinearity)
            print "Dense output shapes"
            print layers.get_output_shape(self.DConvLayers)
            self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout), num_units=layers.get_output_shape(self.DConvLayers)[-1], nonlinearity=self._nonlinearity)
            print layers.get_output_shape(self.DenseLayers)

            self.Conv2DScaleNetLayers = layers.DenseLayer(layers.dropout(self.Conv2DLayers[-1], p=dropout), num_units=2*layers.get_output_shape(self.Conv2DLayers[-1])[-1], nonlinearity=self._nonlinearity)
            self.Conv2DAngleNetLayers = layers.DenseLayer(layers.dropout(self.Conv2DLayers[-1], p=dropout), num_units=2*layers.get_output_shape(self.Conv2DLayers[-1])[-1], nonlinearity=self._nonlinearity)

            for d in range(n_dense-2):
                # Dense Layer 2
                self.DConvLayers = layers.DenseLayer(layers.dropout(self.DConvLayers, p=dropout), num_units=layers.get_output_shape(self.DConvLayers)[-1]/2, nonlinearity=self._nonlinearity)
                self.Conv1DScaleNetLayers = layers.DenseLayer(layers.dropout(self.Conv1DScaleNetLayers, p=dropout), num_units=layers.get_output_shape(self.Conv1DScaleNetLayers)[-1]/2, nonlinearity=self._nonlinearity)
                print "Dense output shapes"
                print layers.get_output_shape(self.DConvLayers)
                self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout), num_units=layers.get_output_shape(self.DConvLayers)[-1], nonlinearity=self._nonlinearity)
                print layers.get_output_shape(self.DenseLayers)
                self.Conv1DAngleNetLayers = layers.DenseLayer(layers.dropout(self.Conv1DAngleNetLayers, p=dropout), num_units=layers.get_output_shape(self.Conv1DAngleNetLayers)[-1]/2, nonlinearity=self._nonlinearity)

                self.Conv2DScaleNetLayers = layers.DenseLayer(layers.dropout(self.Conv2DScaleNetLayers, p=dropout), num_units=layers.get_output_shape(self.Conv2DScaleNetLayers)[-1]/2, nonlinearity=self._nonlinearity)
                self.Conv2DAngleNetLayers = layers.DenseLayer(layers.dropout(self.Conv2DAngleNetLayers, p=dropout), num_units=layers.get_output_shape(self.Conv2DAngleNetLayers)[-1]/2, nonlinearity=self._nonlinearity)

            # Output Layer
            # 1D
            self.DConvLayers = layers.DenseLayer(layers.dropout(self.DConvLayers, p=dropout), num_units=2, nonlinearity=self._nonlinearity)
            self.DenseLayers = layers.DenseLayer(layers.dropout(self.DenseLayers, p=dropout), num_units=2, nonlinearity=self._nonlinearity)
            self.Conv1DScaleNetLayers = layers.DenseLayer(layers.dropout(self.Conv1DScaleNetLayers, p=dropout), num_units=1, nonlinearity=self._nonlinearity)
            self.Conv1DAngleNetLayers = layers.DenseLayer(layers.dropout(self.Conv1DAngleNetLayers, p=dropout), num_units=1, nonlinearity=self._nonlinearity)

            # 2D
            self.Conv2DScaleNetLayers = layers.DenseLayer(layers.dropout(self.Conv2DScaleNetLayers, p=dropout), num_units=1, nonlinearity=self._nonlinearity)
            self.Conv2DAngleNetLayers = layers.DenseLayer(layers.dropout(self.Conv2DAngleNetLayers, p=dropout), num_units=1, nonlinearity=self._nonlinearity)

            # Merge the outputs into a single network
            self.ConvLayers = layers.ConcatLayer([self.Conv1DAngleNetLayers, self.Conv1DScaleNetLayers])
            self.Conv2DLayers = (layers.ConcatLayer([self.Conv2DAngleNetLayers, self.Conv2DScaleNetLayers]))

            if verbose:
                print "Done"
            print

    def train_network(self, num_epochs=1000, stop_err=0.01, d_stable=1e-6, n_stable=10, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, **kwargs):
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
        :param kwargs:      keywords
                            Accepts the following keywords:
                                    verbose: Bool
                                        Whether to print information to the screen.
                                    verbosity: int
                                        Verbosity level of the printed information. Useless unless 'verbose' is True.
                                    plotting: Bool
                                        Whether to plot the training and validation errors of each network after
                                        training has finished.
        :return:
        """
        verbose = kwargs.get('verbose', False)
        verbosity = kwargs.get('verbosity', 3)
        plotting = kwargs.get('plotting', False)

        num_epochs = int(num_epochs)
        self.d_stable = d_stable
        self.n_stable = n_stable

        # Create a loss expression for training
        # 1D
        AE1DPred    = layers.get_output(self.AELayers[-1])
        AE1DLoss    = lasagne.objectives.squared_error(AE1DPred, self.AE_target_var)
        AE1DLoss    = AE1DLoss.mean()

        Conv1DPred  = layers.get_output(self.ConvLayers)
        Conv1DLoss  = lasagne.objectives.squared_error(Conv1DPred, self.target_var)
        Conv1DLoss    = Conv1DLoss.mean()

        DConvPred   = layers.get_output(self.DConvLayers)
        DConvLoss   = lasagne.objectives.squared_error(DConvPred, self.target_var)
        DConvLoss   = DConvLoss.mean()

        DensePred   = layers.get_output(self.DenseLayers)
        DenseLoss   = lasagne.objectives.squared_error(DensePred, self.target_var)
        DenseLoss   = DenseLoss.mean()

        # 2D
        AE2DPred    = layers.get_output(self.AE2DLayers[-1])
        AE2DLoss    = lasagne.objectives.squared_error(AE2DPred, self.AE_target_var2D)
        AE2DLoss    = AE2DLoss.mean()

        Conv2DPred  = layers.get_output(self.Conv2DLayers)
        Conv2DLoss  = lasagne.objectives.squared_error(Conv2DPred, self.target_var)
        Conv2DLoss  = Conv2DLoss.mean()

        # Create an update expression for training
        # 1D
        AE1DParams      = layers.get_all_params(self.AELayers[-1], trainable=True)
        AE1DUpdates     = lasagne.updates.adam(AE1DLoss, AE1DParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon, learning_rate=learning_rate)

        Conv1DParams    = layers.get_all_params(self.ConvLayers, trainable=True)
        Conv1DUpdates   = lasagne.updates.adam(Conv1DLoss, Conv1DParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon, learning_rate=learning_rate)

        DConvParams     = layers.get_all_params(self.DConvLayers, trainable=True)
        DConvUpdates    = lasagne.updates.adam(DConvLoss, DConvParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon, learning_rate=learning_rate)

        DenseParams     = layers.get_all_params(self.DenseLayers, trainable=True)
        DenseUpdates    = lasagne.updates.adam(DenseLoss, DenseParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon, learning_rate=learning_rate)

        # 2D
        AE2DParams          = layers.get_all_params(self.AE2DLayers[-1], trainable=True)
        AE2DUpdates         = lasagne.updates.adam(AE2DLoss, AE2DParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon, learning_rate=learning_rate)

        Conv2DParams        = layers.get_all_params(self.Conv2DLayers, trainable=True)
        Conv2DUpdates       = lasagne.updates.adam(Conv2DLoss, Conv2DParams, beta1=beta_1, beta2=beta_2, epsilon=epsilon, learning_rate=learning_rate)

        # Create a loss expression for validation/testing
        AE1DTest_pred   = layers.get_output(self.AELayers[-1], deterministic=True)
        AE1DTest_loss   = lasagne.objectives.squared_error(AE1DTest_pred, self.AE_target_var)
        AE1DTest_loss   = AE1DTest_loss.mean()

        Conv1DTest_pred = layers.get_output(self.ConvLayers, deterministic=True)
        Conv1DTest_loss = lasagne.objectives.squared_error(Conv1DTest_pred, self.target_var)
        Conv1DTest_loss = Conv1DTest_loss.mean()

        DConvTest_pred  = layers.get_output(self.DConvLayers, deterministic=True)
        DConvTest_loss  = lasagne.objectives.squared_error(DConvTest_pred, self.target_var)
        DConvTest_loss  = DConvTest_loss.mean()

        DenseTest_pred  = layers.get_output(self.DenseLayers, deterministic=True)
        DenseTest_loss  = lasagne.objectives.squared_error(DenseTest_pred, self.target_var)
        DenseTest_loss  = DenseTest_loss.mean()

        # 2D
        AE2DTest_pred     = layers.get_output(self.AE2DLayers[-1], deterministic=True)
        AE2DTest_loss           = lasagne.objectives.squared_error(AE2DTest_pred, self.AE_target_var2D)
        AE2DTest_loss           = AE2DTest_loss.mean()

        Conv2DTest_pred   = layers.get_output(self.Conv2DLayers, deterministic=True)
        Conv2DTest_loss         = lasagne.objectives.squared_error(Conv2DTest_pred, self.target_var)
        Conv2DTest_loss         = Conv2DTest_loss.mean()

        # Compile a second function computing the validation loss:
        AE1DVal_fn          = theano.function([self.input_var, self.AE_target_var], AE1DTest_loss)
        Conv1DVal_fn        = theano.function([self.input_var, self.target_var], Conv1DTest_loss)
        DConvVal_fn         = theano.function([self.input_var, self.target_var], DConvTest_loss)
        DenseVal_fn         = theano.function([self.input_var, self.target_var], DenseTest_loss)
        AE2DVal_fn          = theano.function([self.input_var2D, self.AE_target_var2D], AE2DTest_loss)
        Conv2DVal_fn        = theano.function([self.input_var2D, self.target_var], Conv2DTest_loss)

        # Create a training function
        # 1D
        AE1DTrainer         = theano.function([self.input_var, self.AE_target_var], AE1DLoss, updates=AE1DUpdates)
        Conv1DTrainer       = theano.function([self.input_var, self.target_var], Conv1DLoss, updates=Conv1DUpdates)
        DConvTrainer        = theano.function([self.input_var, self.target_var], DConvLoss, updates=DConvUpdates)
        DenseTrainer        = theano.function([self.input_var, self.target_var], DenseLoss, updates=DenseUpdates)

        # 2D
        AE2DTrainer         = theano.function([self.input_var2D, self.AE_target_var2D], AE2DLoss, updates=AE2DUpdates)
        Conv2DTrainer       = theano.function([self.input_var2D, self.target_var], Conv2DLoss, updates=Conv2DUpdates)

        # Initialize Training Variables
        AE1DErr         = []
        AE2DErr         = []
        Conv1DErr       = []
        Conv2DErr       = []
        DConvErr        = []
        DenseErr        = []

        AE1DValErr      = []
        AE2DValErr      = []
        Conv1DValErr    = []
        Conv2DValErr    = []
        DConvValErr     = []
        DenseValErr     = []

        AE1DTrain_time      = []
        AE2DTrain_time      = []
        Conv1DTrain_time    = []
        Conv2DTrain_time    = []
        DConvTrain_time     = []
        DenseTrain_time     = []

        AE1Dflag            = False
        AE2Dflag            = False
        Conv1Dflag          = False
        Conv2Dflag          = False
        DConvflag           = False
        Denseflag           = False

        epochs = dict()

        AETrain_time        = time.time()
        for i in range(num_epochs):
            if verbose and verbosity >=2:
                print "##"*50
                print "Pretraining Epoch %r" %i
                print "--" * 50
            if not AE1Dflag:
                if verbose and verbosity >= 2:
                    print "Training 1D autoencoder..."
                t = time.time()
                AE1DErr.append(AE1DTrainer(self.X_train, self.X_train))
                AE1DTrain_time.append(time.time()-t)
                AE1DValErr.append(AE1DVal_fn(self.X_val, self.X_val))
                if AE1DErr[-1] < stop_err:
                    if verbose and verbosity >= 1:
                        print "1-D training converged."
                    epochs.update({"AE1D": i+1})
                    AE1Dflag    = True
                elif self.stablecheck(AE1DErr):
                    if verbose and verbosity >=1:
                        print "1-D error converged before training completed."
                    epochs.update({"AE1D": i+1})
                    AE1Dflag    = True

            if not AE2Dflag:
                if verbose and verbosity >= 2:
                    print "Training 2D autoencoder..."
                t = time.time()
                AE2DErr.append(AE2DTrainer(self.X_train2D, self.X_train2D))
                AE2DTrain_time.append(time.time()-t)
                AE2DValErr.append(AE2DVal_fn(self.X_val2D, self.X_val2D))
                if AE2DErr[-1] < stop_err:
                    if verbose and verbosity >= 1:
                        print "2-D training converged."
                    epochs.update({"AE2D": i+1})
                    AE2Dflag    = True
                elif self.stablecheck(AE2DErr):
                    if verbose and verbosity >= 1:
                        print "2-D error converged before training completed."
                    epochs.update({"AE2D": i+1})
                    AE2Dflag    = True
            if AE1Dflag and AE2Dflag:
                break
            if verbose: print
        AETrain_time = time.time() - AETrain_time
        if not AE1Dflag:
            epochs.update({"AE1D": i+1})
        if not AE2Dflag:
            epochs.update({"AE2D": i+1})
        if verbose and verbosity >= 1:
            print "##" * 50
            print "Pretraining Completed"
            print "Total Time: {:.3f}s".format(AETrain_time)
            print "--" * 50
            print "1-D Autoencoder Error: ", AE1DErr[-1]
            print "1-D Autoencoder Validation Error: ", AE1DValErr[-1]
            print "Training Time: {:.3f}s".format(np.sum(AE1DTrain_time))
            print "Training Epochs: ", epochs["AE1D"]
            print "--"*50
            print "2-D Autoencoder Error: ", AE2DErr[-1]
            print "2-D Autoencoder Validation Error: ", AE2DValErr[-1]
            print "Training Time: {:.3f}s".format(np.sum(AE2DTrain_time))
            print "Training Epochs: ", epochs["AE2D"]
            print "##"*50
            print "##"*50
            print
            print

        NetTrain_time       = time.time()
        for i in range(num_epochs):
            if verbose and verbosity >= 2:
                print "##"*50
                print "Training Epoch %r" % i
                print "--"*50
            if not Conv1Dflag:
                if verbose and verbosity >= 2:
                    print "Training pretrained 1-D convolutional net..."
                t = time.time()
                Conv1DErr.append(Conv1DTrainer(self.X_train, self.y_train))
                Conv1DTrain_time.append(time.time()-t)
                Conv1DValErr.append(Conv1DVal_fn(self.X_val, self.y_val))
                if Conv1DErr[-1] < stop_err:
                    if verbose and verbosity >= 1:
                        print "1-D training converged."
                    epochs.update({"Conv1D": i+1})
                    Conv1Dflag  = True
                elif self.stablecheck(Conv1DErr):
                    if verbose and verbosity >= 1:
                        print "1-D error converged before training completed."
                    epochs.update({"Conv1D": i+1})
                    Conv1Dflag  = True
            if not Conv2Dflag:
                if verbose and verbosity >=2:
                    print "Training pretrained 2-D convolutional net..."
                t   = time.time()
                Conv2DErr.append(Conv2DTrainer(self.X_train2D, self.y_train))
                Conv2DTrain_time.append(time.time()-t)
                Conv2DValErr.append(Conv2DVal_fn(self.X_val2D, self.y_val))
                if Conv2DErr[-1] < stop_err:
                    if verbose and verbosity >= 1:
                        print "2-D training converged."
                    epochs.update({"Conv2D": i+1})
                    Conv2Dflag  = True
                elif self.stablecheck(Conv2DErr):
                    if verbose and verbosity >= 1:
                        print "2-D error converged before training completed."
                    epochs.update({"Conv2D": i+1})
                    Conv2Dflag  = True
            if not DConvflag:
                if verbose and verbosity >= 2:
                    print "Training \"Dumb\" Convolutional Net..."
                t   = time.time()
                DConvErr.append(DConvTrainer(self.X_train, self.y_train))
                DConvTrain_time.append(time.time()-t)
                DConvValErr.append(DConvVal_fn(self.X_val, self.y_val))
                if DConvErr[-1] < stop_err:
                    if verbose and verbosity >= 1:
                        print "1-D non-pretrained training converged."
                    epochs.update({"DConv": i+1})
                    DConvflag   = True
                elif self.stablecheck(DConvErr):
                    if verbose and verbosity >= 1:
                        print "1-D non-pretrained error converged before training completed."
                    epochs.update({"DConv": i+1})
                    DConvflag   = True
            if not Denseflag:
                if verbose and verbosity >= 2:
                    print "Training MLP..."
                t   = time.time()
                DenseErr.append(DenseTrainer(self.X_train, self.y_train))
                DenseTrain_time.append(time.time()-t)
                DenseValErr.append(DenseVal_fn(self.X_val, self.y_val))
                if DenseErr[-1] < stop_err:
                    if verbose and verbosity >= 1:
                        print "Fully connected training converged."
                    epochs.update({"Dense": i+1})
                    Denseflag   = True
                elif self.stablecheck(DenseErr):
                    if verbose and verbosity >= 1:
                        print "Fully connected error converged before training completed."
                    epochs.update({"Dense": i+1})
                    Denseflag   = True
            if Conv1Dflag and Conv2Dflag and DConvflag and Denseflag:
                break
            if verbose:
                print

        NetTrain_time = time.time() - NetTrain_time
        if not Conv1Dflag:
            epochs.update({"Conv1D": i+1})
        if not Conv2Dflag:
            epochs.update({"Conv2D": i+1})
        if not DConvflag:
            epochs.update({"DConv": i+1})
        if not Denseflag:
            epochs.update({"Dense": i+1})
        self.epochs = epochs
        if verbose and verbosity >= 1:
            print
            print "##"*50
            print "Training Completed: %r Convolutional Layers, %r Dense Layers" % (self._n_conv, self._n_dense)
            print "Total Time: {:.3f}s".format(NetTrain_time)
            print "%r 1-D filters, %r 2-D filters" % (self._n_1Dfilters, self._n_2Dfilters)
            print "Nonlinearity: %r" %(self.nonlinearity)
            print "Frozen Weights: %r" %(self._frozen)
            print "--"*50
            print "1-D Dual Output Error: ", Conv1DErr[-1]
            print "Training Time: {:.3f}s".format(np.sum(Conv1DTrain_time))
            print "Training Epochs: ", epochs["Conv1D"]
            print "--"*50
            print "2-D Dual Output Error: ", Conv2DErr[-1]
            print "Training Time: {:.3f}s".format(np.sum(Conv1DTrain_time))
            print "Training Epochs: ", epochs["Conv2D"]
            print "--"*50
            print "1-D Dumb Dual Output Error: ", DConvErr[-1]
            print "Training Time: {:.3f}s".format(np.sum(DConvTrain_time))
            print "Training Epochs: ", epochs["DConv"]
            print "--"*50
            print "MLP Output Error: ", DenseErr[-1]
            print "Training Time: {:.3f}s".format(np.sum(DenseTrain_time))
            print "Training Epochs: ", epochs["Dense"]
            print "##"*50
            print "##"*50
            print "##"*50

        if plotting:
            plt.title("MLP and 1-D Unpretrained Convolutional Network Training and Validation Error")
            plt.plot(DConvErr, 'g', label='Convolutional, Training')
            plt.plot(DConvValErr, 'b', label='Convolutional, Validation')
            plt.plot(DenseErr, 'k', label='MLP, Training')
            plt.plot(DenseValErr, 'r', label='MLP, Validation')
            plt.xlabel("Epoch")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.show()

            plt.title("1-D Convolutional Network Training and Validation Error: %r filters" %(self._n_1Dfilters))
            plt.plot(Conv1DErr, 'g', label='Training')
            plt.plot(Conv1DValErr, 'b', label='Validation')
            plt.xlabel("Epoch")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.show()

            plt.title("2-D Convolutional Network Training and Validation Error: %r filters" %(self._n_2Dfilters))
            plt.plot(Conv2DErr, 'g', label='Training')
            plt.plot(Conv2DValErr, 'b', label='Validation')
            plt.xlabel("Epoch")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.show()

        self._nets = {0: theano.function([self.input_var], layers.get_output(self.ConvLayers, deterministic=True), name="1D Convolutional Network"),
                      3: theano.function([self.input_var2D], layers.get_output(self.Conv2DLayers, deterministic=True), name="2D Convolutional Network"),
                      5: theano.function([self.input_var], layers.get_output(self.DenseLayers, deterministic=True), name="MLP Network"),
                      6: theano.function([self.input_var], layers.get_output(self.DConvLayers, deterministic=True), name="1D Convolutional Network Sans Pretraining")
                      }
        data2, target = self._data2D[:2], self._targets[:2]
        print "Test"
        print "Expected: ", target
        predictions, networks, errors = self.solve(input2D=self.shape_data2D(data2, channel_axis=1, batch_axis=0, batched=True), targets=target, network_ids=3)
        print "Predictions: ", predictions
        print "Networks: ", networks
        print "Errors: ", errors

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

    def shape_data2D(self, data2D, channel_axis=-1, batched=False, batch_axis=0):
        """
        :param data2D: ndarray
                            Image data to be shaped. If the pixels are ints, scale them to floats in [0, 1].
        :param channel_axis: int, optional
                            Axis which stores the channel width. By default this value is -1 in the case of the last
                            axis denoting the number of channels.
        :param batched: Bool, optional
                            Whether or not the incoming data is a singleton or a batch. If incoming data has 4
                            dimensions it is assumed to be batched
        :param batch_axis: int, optional
                            Integer specifying which axis contains the batch size.
        :return: ndarray
                            Reshaped data whose dimensions are (batch size, number of channels, height, width)
        """
        assert len(data2D.shape) <= 4, "Image data has too many dimensions. Unsure how to reshape"
        if data2D.dtype == int:
            data2D = data2D/255.
        elif data2D.dtype == float:
            pass
        else:
            raise TypeError("Image data type must be float or int. Instead got %r" %(data2D.dtype))
        if len(data2D.shape) is 2:
            reshaped = np.zeros((1, 1, data2D.shape[0], data2D.shape[1]), dtype=float)
            reshaped[0, 0, :, :] = data2D
        elif len(data2D.shape) is 3:
            if not batched:
                axes = []
                for i in range(3):
                    if i is not (channel_axis%3):
                        axes.append(i)
                data2D.reshape((data2D.shape[channel_axis], data2D.shape[axes[0]], data2D.shape[axes[1]]))
                reshaped = np.zeros((1, data2D.shape[0], data2D.shape[1], data2D.shape[2]), dtype=float)
                reshaped[0, :, :, :] = data2D
            else:
                axes = []
                for i in range(3):
                    if i is not (batch_axis%3):
                        axes.append(i)
                data2D.reshape((data2D.shape[batch_axis], data2D.shape[axes[0]], data2D.shape[axes[1]]))
                reshaped = np.zeros((data2D.shape[0], 1, data2D.shape[1], data2D.shape[2]), dtype=float)
                reshaped[:, 0, :, :] = data2D
        elif len(data2D.shape) is 4:
            axes = []
            for i in range(4):
                if i is not (channel_axis%4) and i is not (batch_axis%4):
                        axes.append(i)
            data2D.reshape((data2D.shape[batch_axis], data2D.shape[channel_axis], data2D.shape[axes[0]], data2D.shape[axes[1]]))
            reshaped = data2D

        m, n = np.max(reshaped), np.min(reshaped)
        if m > 1.:
            raise ValueError("Image pixel values are outside accepted range. Max value %r must be less than 1." %(m))
        elif n < 0.:
            raise ValueError("Image pixel values are outside accepted range. Min value %r must be nonnegative" %(n))
        return reshaped

    def shape_data1D(self, data1D, batched=False, batch_axis=0):
        """
        :param data1D: ndarray
                            Curvature data to be reshaped.
        :param batched: Bool, optional
                            Whether or not the incoming data is a singleton or a batch. If incoming data has 3
                            dimensions it is assumed to be batched.
        :param batch_axis: int, optional
                            Integer specifying which axis contains the batch size.
        :return: ndarray
                            Reshaped data whose dimensions are (batch size, number of channels, data points)
        """
        assert len(data1D.shape) <= 3, "Data has too many dimensions. Unsure how to reshape."
        if len(data1D.shape) is 1:
            reshaped = np.zeros((1, 1, data1D.shape[0]), dtype=float)
            reshaped[0,0, :] = data1D
        elif len(data1D.shape) is 2:
            if not batched:
                for i in range(2):
                    if i is not channel_axis:
                        axes = [i]
                reshaped = np.zeros((1, data1D.shape[channel_axis], data1D.shape[axes[0]]), dtype=float)
                reshaped[0, :, :] = data1D
            else:
                for i in range(2):
                    if i is not (batch_axis%2):
                        axes = [i]
                reshaped = np.zeros((data1D.shape[batch_axis], 1, data1D.shape[axes[0]]), dtype=float)
                reshaped[:, 0, :] = data1D
        elif len(data1D.shape) is 3:
                axes = []
                for i in range(3):
                    if i is not (channel_axis % 3) and i is not (batch_axis%3):
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
        x[0] = x[0]**2.
        x[1] = (x[1] + .5)*2.*np.pi
        x[1] = x[1]%(2.*np.pi)
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
            error = (np.abs(x[0]-y[0]), np.abs(x[1]-y[1]))
        else:
            error = (np.abs(x[0]-y[0]-epsilon)/np.abs(y[0]+epsilon), np.abs(x[1]-y[1]-epsilon)/np.abs(y[1])+epsilon)
        return error


    def solve(self, input1D=None, input2D=None, targets=None, network_ids=None, timing=False, batch_process=False, **kwargs):
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
                                                0: Dual-output 1D convolutional network
                                                1: Scale-output 1D convolutional network
                                                2: Angle-output 1D convolutional network
                                                3: Dual-output 2D convolutional network
                                                4: Scale-ouput 2D convolutional network
                                                5: Angle-ouput 2D convolutional network
                                                6: Non-pretrained 1D convolutional network
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
            assert targets.shape[-1] == 2, "Targets shape not in output space. Expected 2 targets per sample, instead got $r per sample." % (targets.shape[-1])
            relative = kwargs.get("relative", True)
            epsilon = kwargs.get("epsilon", 1e-8)

        if input1D is None and input2D is None:
            raise UnboundLocalError("No input specified.")
        elif input2D is None:
            assert len(input1D.shape)== 3, "ShapeError: Improper number of dimensions for 1D input. Expected 3 with dimensions (batch size, 1, data points per sample) and got %r" %(input1D.shape)
            assert input1D.shape[-1] == self._input1D_size, "1D ShapeError: Expected %r inputs per sample. Got %r instead." %(self._input1D_size, input1D.shape[-1])
            if targets is not None:
                assert input1D.shape[0] == targets.shape[0], "1D input batch size (%r) does not match expected batch size (%r)." % (input1D.shape[0], targets.shape[0])
            batch_size = input1D.shape[0]
        elif input1D is None:
            assert len(input2D.shape)== 4, "ShapeError: Improper number of dimensions for image input. Expected 4 with dimensions (batch size, number of channels, height, width) and got %r" %(input2D.shape)
            assert input2D.shape[1:] == self._input2D_shape, "Image ShapeError: Expected image input with shape %r, instead got input with shape %r." %(self._input2D_shape, input2D.shape[1:])
            if targets is not None:
                assert input2D.shape[0] == targets.shape[0], "Image input batch size (%r) does not match expected batch size (%r)." % (input2D.shape[0], targets.shape[0])
            batch_size = input2D.shape[0]
        else:
            assert len(input1D.shape) == 3, "ShapeError: Improper number of dimensions for 1D input. Expected 3 with dimensions (batch size, 1, data points per sample) and got %r" % (input1D.shape)
            assert input1D.shape[-1] == self._input1D_size, "1D ShapeError: Expected %r inputs per sample. Got %r instead." % (self._input1D_size, input1D.shape[-1])
            assert len(input2D.shape) == 4, "ShapeError: Improper number of dimensions for image input. Expected 4 with dimensions (batch size, number of channels, height, width) and got %r" % (input2D.shape)
            assert input2D.shape[1:] == self._input2D_shape, "Image ShapeError: Expected image input with shape %r, instead got input with shape %r." % (self._input2D_shape, input2D.shape[1:])
            assert input1D.shape[0] == input2D.shape[0], "1-D batch size (%r) and 2-D batch size (%r) do not match." % (input1D.shape[0], input2D.shape[0])
            batch_size = input1D.shape[0]
            if targets is not None:
                assert targets.shape[0] == batch_size, "Targets batch size (%r) does not match batch size (%r)." % (targets.shape[0], batch_size)

        if network_ids is None:
            # network_ids = [i for i in range(7)]
            network_ids = [3*i for i in range(3)]
        elif type(network_ids) is int:
            network_ids = [network_ids]
        elif type(network_ids) is np.ndarray:
            network_ids = list(network_ids.flatten())
        else:
            # should raise an error if network_ids is not iterable, which is the necessary functionality
            iter(network_ids)
        networks = [self._nets[id].name for id in network_ids]
        for i in range(len(network_ids)):
            if i not in range(7):
                raise ValueError("Invalid network ID supplied. Must be an integer between 0 and 6. The invalid ID is %r" %(i))

        if input1D is None and 0 in network_ids or 6 in network_ids or 5 in network_ids:
            warnings.warn("Asked to return data from 1-D convolutional networks but no 1-D data was given. Ignoring invalid requests.", Warning)
            while 0 in network_ids or 6 in network_ids:
                try:
                    network_ids.remove(0)
                    network_ids.remove(6)
                except ValueError:
                    pass
        elif input2D is None and 3 in network_ids:
            warnings.warn("Asked to return data from 2-D convolutional networks but no image data was given. Ignoring invalid requests.", Warning)
            while 3 in network_ids:
                try:
                    network_ids.remove(3)
                except ValueError:
                    pass
        assert len(network_ids) is not 0, "No valid network id's specified for the input data types."


        predictions = np.empty((batch_size, len(network_ids), 2))
        if targets is not None:
            errors  = np.empty((batch_size, len(network_ids), 2))

        if verbose:
            print "Solving..."
        if timing:
            start = time.time()
            if verbosity >=2:
                times = [start]
        if batch_process:
            for id in range(len(network_ids)):
                if network_ids[id] in [0, 6]:
                    batch_prediction = self._nets[network_ids[id]](input1D)
                    predictions[:, id, :] = [self.to_output(sample) for sample in batch_prediction]
                elif network_ids[id] in [3]:
                    batch_prediction = self._nets[network_ids[id]](input2D)
                    predictions[:, id, :] = [self.to_output(sample) for sample in batch_prediction]
            if targets is not None:
                for batch in range(batch_size):
                    errors[batch, id, :] = self.get_error(prediction[batch], targets[batch], relative, epsilon)
        else:
            for batch in range(batch_size):
                for id in range(len(network_ids)):
                    if network_ids[id] in [0, 6]:
                        prediction = self.to_output(self._nets[network_ids[id]](input1D)[0])
                        predictions[batch, id, :] = prediction
                    elif network_ids[id] in [3]:
                        prediction = self.to_output(self._nets[network_ids[id]](input2D)[0])
                        predictions[batch, id, :] = prediction
                    if timing:
                        if verbose and verbosity >= 2:
                            times.append(time.time())
                            print "Sample {} solution time: {:.3f}s".format(batch, times[-1] - times[-2])
                    if targets is not None:
                        errors[batch, id, :] = self.get_error(prediction, targets[batch], relative, epsilon)
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
                print "--"*50
                print "Total Time: {:.3f}s".format(stop)
                if verbosity >= 2:
                    times = np.diff(times)
                    print "Mean time: {:.3f}s".format(np.mean(times))
                    print "Median time: {:.3f}s".format(np.median(times))
                    print "Max time: {:.3f}s".format(np.max(times))
                    print "Min time: {:.3f}s".format(np.min(times))
                    print "Standard deviation: {:.3f}s".format(np.std(times))
                    print "##"*50
                    print

        if targets is None:
            return predictions, networks
        else:
            return predictions, networks, errors

        # print "Predictions"
        # answer = a3[0]
        # a4 = a4[0]
        # answer[0] = 2.**answer[0]
        # a4[0] = 2.**a4[0]
        # aS1D[0] = 2.**aS1D[0]
        # aS2D[0] = 2.**aS2D[0]
        # # answer[1], answer[3] = (answer[1]+.5)*(2.*np.pi), (answer[3]+.5)*(2.*np.pi)
        # answer[1] = (answer[1]+.5)*(2.*np.pi)
        # a4[1] = (a4[1]+.5)*(2.*np.pi)
        # aA1D[0] = (aA1D[0]+.5)*(2.*np.pi)
        # aA2D[0] = (aA2D[0]+.5)*(2.*np.pi)
        #
        # a5 = a5[0]
        # print "1D Scale Isolated: ", aS1D[0]
        # print "Error: ", (aS1D[0]-expected[0])/expected[0]
        # print "2D Scale Isolated: ", aS2D[0]
        # print "Error: ", (aS2D[0]-expected[0])/expected[0]
        # print "1D Angle Isolated: ", aA1D[0]
        # print "Error: ", (aA1D[0]-expected[1])/expected[1]
        # print "2D Angle Isolated: ", aA2D[0]
        # print "Error: ", (aA2D[0]-expected[1])/expected[1]
        # print "Pretrained: ", answer
        # print "Error: ", [(answer[i]-expected[i])/expected[i] for i in range(len(expected))]
        # print "Pretrained 2D: ", a5
        # print "Error: ", [(a5[i]-expected[i])/expected[i] for i in range(len(expected))]
        # print "Dumb: ", a4
        # print "Error: ", [(a4[i]-expected[i])/expected[i] for i in range(len(expected))]
        # print
        # return answer, a4

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

class Contour(object):
    def __init__(self, contour, sigma=1., scale=1., rescale=False, **kwargs):
        self.contour = contour
        # print 'Contour init'
        # print np.amax(contour, 0)
        self.rows, self._cols = zip(*contour)
        self.sigma = sigma
        self.scale = scale
        # print "setting cimage in contour init"
        self.cimage = self.contourtoimg(rescale=rescale).astype(bool)
        # plt.imshow(self.cimage)
        # plt.show()
        self.curvature(sigma=self._sigma)
        # print "after curvature"
        # print np.amax(self.contour, 0)
        # print
    def __iter__(self):
        return iter(self._contour)
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma
    @property
    def rows(self):
        return self._rows
    @rows.setter
    def rows(self, rows):
        self._rows = rows
        self._contour = zip(rows, self.cols)
    @property
    def cols(self):
        return self._cols
    @cols.setter
    def cols(self, cols):
        self._cols = cols
        self._contour = zip(self._rows, cols)
    @property
    def contour(self):
        return self._contour
    @contour.setter
    def contour(self, contour):
        self._contour = contour
        self._rows, self._cols = zip(*contour)
    @property
    def curvatures(self):
        return self._curvatures
    @property
    def centroid(self):
        return self._centroid
    @centroid.setter
    def centroid(self, value):
        self._centroid = (value[0], value[1])
    @property
    def smooth_contour(self):
        return self._smooth_contour
    @property
    def angles(self):
        return self._angles
    @angles.setter
    def angles(self, angles):
        self._angles = list(angles)
    @property
    def __len__(self):
        return len(self._contour)
    def getangles(self, smoothed=False, sigma=1.):
        '''Calculates the angle from the centroid to each edge pixel'''
        '''Returns the list of angles to pixel locations for the leaf at index. If no index is specified, returns a list containing the lists of angles for each leaf'''
        self.findcentroid()
        self.getradii(smoothed=smoothed, sigma=sigma)
        if smoothed:
            length=len(self._smooth_rows)
            ts = np.arcsin([float(self._centroid[0] - self._smooth_rows[_])/self._radii[_] for _ in range(length)])
            tc = np.arccos([float(self._smooth_cols[_] - self._centroid[1])/self._radii[_] for _ in range(length)])
        else:
            length = len(self.contour)
            ts = np.arcsin([float(self._centroid[0] - self._rows[_])/self._radii[_] for _ in range(length)])
            tc = np.arccos([float(self._cols[_] - self._centroid[1])/self._radii[_] for _ in range(length)])
        thetas = []
        for j in range(length):
            if ts[j]<0 and tc[j]>np.pi/2.:
                thetas.append(2.*np.pi - tc[j])
            elif ts[j]<0:
                thetas.append(2.*np.pi + ts[j])
            else:
                thetas.append(tc[j])
        self._angles = thetas
        return self._angles
    @property
    def radii(self):
        return self._radii
    @radii.setter
    def radii(self, radii):
        self._radii= list(radii)
    @property
    def scale(self):
        return self._scale
    @scale.setter
    def scale(self, scale):
        self._scale = scale
    @property
    def cimage(self):
        return self._cimage
    @cimage.setter
    def cimage(self, img):
        self._cimage = img
    @cimage.deleter
    def cimage(self):
        del self._cimage
    def curvature(self, sigma=1., **kwargs):
        '''Computes the curvature at each point along the edge'''
        self._sigma = sigma

        x   = [float(x) for x in self._cols]
        y   = [float(y) for y in self._rows]

        xu  = gaussian_filter1d(x, self._sigma, order=1, mode='wrap')
        yu  = gaussian_filter1d(y, self._sigma, order=1, mode='wrap')

        xuu = gaussian_filter1d(xu, self._sigma, order=1, mode='wrap')
        yuu = gaussian_filter1d(yu, self._sigma, order=1, mode='wrap')

        k = [(xu[i]*yuu[i]-yu[i]*xuu[i])/np.power(xu[i]**2.+ yu[i]**2., 1.5) for i in range(len(xu))]
        self._curvatures = k

        if kwargs.get('visualize', False):
            plt.plot(k)
            plt.show()
        return k
    def extractpoint(self, theta, **kwargs):
        '''Finds the index of the point with an angle closest to theta'''
        self.getangles(**kwargs)
        diff = list(np.abs(self.angles[:]-theta))
        return diff.index(np.min(diff))
    def findcentroid(self):
        self._centroid = (int(np.mean(self._rows)), int(np.mean(self._cols)))
        return self._centroid
    def getradii(self, smoothed=False, sigma=1.):
        self.smooth(sigma)
        if smoothed:
            self._radii = [np.sqrt((self._smooth_rows[_]-float(self._centroid[0]))**2.+(self._smooth_cols[_]-float(self._centroid[1]))**2.) for _ in range(self._smooth_len)]
        else:
            self._radii = [np.sqrt(float((self._rows[_]-float(self._centroid[0]))**2.+(self._cols[_]-float(self._centroid[1]))**2.)) for _ in range(len(self._contour))]
        return self._radii
    def orient(self, smoothed=True, sigma=1., **kwargs):
        # print "orient"
        # print "using sigma=", sigma
        # print "sigma=self.sigma", sigma==self.sigma
        visualize = kwargs.get('visualize', False)
        rescale = kwargs.get('rescale', True)
        resize = kwargs.get('resize', True)
        preserve_range = kwargs.get('preserve_range', True)
        if visualize:
            print "Before Orientation"
            cols = int(1.1*np.max(self._cols))
            rows = int(1.1*np.max(self._rows))
            shape = (rows, cols)
            img = np.zeros(shape, bool)

            for c in self._contour:
                img[c] = True
            plt.imshow(img)
            plt.show()
        self.curvature(sigma=sigma, **kwargs)
        # Retrieve index of highest curvature point. Rotate curve
        index = self._curvatures.index(np.max(self._curvatures))
        angles = self.getangles(smoothed=smoothed, sigma=sigma)
        angle = angles[index]
        length = self.__len__
        # print "angle offset"
        # print angle
        angles = angles - angle + np.pi/2.
        self._angles = angles
        # print "angles max/min"
        # print np.max(angles)
        # print np.min(angles)
        # print
        curve = [(self._radii[_]*np.sin(self._angles[_]), self._radii[_]*np.cos(self._angles[_])) for _ in range(length)]
        dimspan = np.ptp(curve, 0)
        dimmax = np.amax(curve, 0)
        centroid = (dimmax[0] + dimspan[0]/2, dimmax[1] + dimspan[1]/2)
        curve = [(c[0] + centroid[0], c[1] + centroid[1]) for c in curve]


        self.cimage = tf.rotate(self.cimage, -angle*180./np.pi+90., resize=True).astype(bool)
        # print "Still 64x64?: ", self.cimage.shape[0] == 64 and self.cimage.shape[1] == 64
        # print self.cimage.shape
        self.cimage, scale, contour = iso_leaf(self.cimage, True)
        # self.cimage = tf.rotate(self.cimage, angle*180./np.pi, resize=True).astype(bool)
        # plt.imshow(img)
        # plt.show()
        # print "Resizing to preserve 64x64"
        # print img.shape
        self.image = tf.rotate(self.image, -angle*180./np.pi+90., resize=True)
        self.image = imresize(self.image, scale, interp="bicubic")
        self.scale *= scale
        # self.cimage = rotate(self.cimage, -angle*180./np.pi).astype(bool)
        # print "cimage after reorientation"
        # plt.imshow(self.cimage)
        # plt.show()
        # np.pad(self.cimage)
        curve1 = parametrize(self.cimage)
        # print "s"
        # self.contourtoimg(rescale=True)

        # Scale contour by scaling factor. Set 0 index to be top of leaflet
        length = len(curve)
        curve = [curve[(index+_)%length] for _ in range(length)]
        # print "Oriented cimage"
        # plt.imshow(self.cimage)
        # plt.show()
        # print "curve 1 from contourtoimg"
        c1img = np.zeros_like(self.cimage)
        for c in curve1:
            c1img[c] = True
        # img = self.contourtoimg(curve1, rescale=True)
        # plt.imshow(c1img)
        # plt.show()
        #
        # try:
        #     print "Int of centroids"
        #     curve = [(int(np.round(c[0])), int(np.round(c[1]))) for c in curve]
        #     img = self.contourtoimg(curve, rescale=True)
        #     plt.imshow(img)
        #     plt.show()
        #     print "contour to img of curve1"
        #     img = self.contourtoimg(curve1, rescale=True)
        #     plt.imshow(img)
        #     plt.show()
        #     print
        # except ValueError:
        #     print "ValueError"
        #     img = self.contourtoimg(curve1, rescale=False)
        #     plt.imshow(img)
        #     plt.show()
        #     img = self.contourtoimg(curve, rescale=False)
        #     plt.imshow(img)
        #     plt.show()
        self.cimage = c1img
        # print 'orient'
        # print np.amax(self._contour)

        self.getangles(smoothed=smoothed, sigma=sigma)
        self.smooth(sigma=sigma)
        self.curvature(sigma=sigma)

        if visualize:
            print "After Orientation"
            shape = (2*self.centroid[0], 2*self.centroid[1])
            img = np.zeros(shape, bool)
            for c in self._contour:
                img[c] = True
            plt.imshow(img)
            plt.show()
            print '-'*40
            print
    def plot(self, smoothed=False):
        if not smoothed:
            rowmax, colmax = np.max(self.rows), np.max(self.cols)
        else:
            rowmax, colmax = np.max(self.smooth_rows), np.max(self.smooth_cols)

        img = np.zeros((rowmax+1, colmax+1), bool)
        for c in self._contour:
            img[c] = True
        plt.imshow(img)
        plt.show()

    def smooth(self, sigma=1.):
        # self._sigma = kwargs.get('sigma', 1.)
        self._sigma = sigma
        self._smooth_rows = tuple(gaussian_filter1d(self._rows, self._sigma, order = 0, mode='wrap'))
        self._smooth_cols = tuple(gaussian_filter1d(self._cols, self._sigma, order = 0, mode='wrap'))
        self._smooth_contour = zip(self._smooth_rows, self._smooth_cols)
        self._smooth_len = len(self._smooth_contour)
        return self._smooth_contour
    def contourtoimg(self, contour=None, shape=(64, 64), rescale=False):
        """Generate an image from a contour.
                contour: the contour from which the image is to be generated. If none is specified then use self.contour
                shape: tuple of ints specifying the image shape
                rescale: rescale the image to shape"""
        if contour is None:
            cont = self.contour
        else:
            cont = contour

        ubounds = np.amax(cont, 0)

        img = np.zeros(ubounds[:]+1, bool)

        if len(cont) > 0 or min(np.ptp(cont, 0))>1:
            for c in cont:
                img[c] = True
        else:
            raise ValueError('contour has no length')
        img, scale, cont = iso_leaf(img, True, square_length=max(shape), rescale=rescale)
        cont = parametrize(img)
        img = np.zeros_like(img)
        for c in cont:
            img[c] = True
        if contour is None:
            self.cimage = img
            self.contour = cont
        return img.astype(bool)

def image2gray(image):
    img = np.zeros(image.shape[:2], np.uint8)
    for (i, j, k), value in np.ndenumerate(image):
        img[i, j] = int(np.mean(value))
    return img

class Leaf(Contour):
    def __init__(self, image, contour=None, img_from_contour=False, orient=True, rescale=False, sigma=1., **kwargs):
        smoothed = kwargs.get('smoothed', False)
        self.__name__ = kwargs.get('name', 'Leaf')
        if contour is None:
            if type(image) is list:
                super(Leaf, self).__init__(image, sigma=sigma, rescale=rescale, name=self.__name__)
            else:
                if len(image.shape) == 3:
                    if np.issubdtype(np.max(image), int):
                        print "NumPy subdtype int"
                        print np.issubdtype(np.max(image, int))
                        self.color_image = image/256.
                    elif np.issubdtype(np.max(image), float):
                        print "NumPy subdtype float"
                        print np.issubdtype(np.max(image, float))
                        self.color_image = image
                    image = image2gray(image)
                else:
                    self.color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
                    self.color_image[:, :, 0] = image/256.
                    self.color_image[:, :, 1] = image/256.
                    self.color_image[:, :, 2] = image/256.
                img = image.astype(bool)
                plt.imshow(img.astype(bool))
                plt.show()
                super(Leaf, self).__init__(parametrize(img), sigma=sigma, rescale=rescale, name=self.__name__)
                self.image = image
        else:
            if len(image.shape) == 3:
                self.color_image = image/256.
                image = image2gray(image)
            else:
                self.color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
                self.color_image[:, :, 0] = image/256.
                self.color_image[:, :, 1] = image/256.
                self.color_image[:, :, 2] = image/256.
            self.image = image
            super(Leaf, self).__init__(contour, sigma=sigma, rescale=rescale, name=self.__name__)

        if img_from_contour:
            # Create images from the given contour
            self.image = self.contourtoimg(contour, shape=kwargs.get('shape', (64, 64)), rescale=rescale)
            self.color_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.float32)
            self.color_image[:, :, 0] = self.image
            self.color_image[:, :, 1] = self.image
            self.color_image[:, :, 2] = self.image

        self.smooth(sigma=sigma)
        self.curvature(**kwargs)
        self.getangles(smoothed=smoothed, sigma=sigma)

        if orient:
            self.orient(smoothed=True, sigma=sigma, **kwargs)
        self.data = list()
        self.curvature(**kwargs)
        self.getangles(smoothed=smoothed, sigma=sigma)
    def __iter__(self):
        return iter(self.contour)
    @property
    def __len__(self):
        return len(self.contour)

    @property
    def image(self):
        return self._image
    @image.setter
    def image(self, img):
        self._image = img
    def distance(self, p0, p1):
        '''Euclidean distance between pixel 0 and pixel 1'''
        return np.sqrt(float(p0[0]-p1[0])**2.+float(p0[1]-p1[1])**2.)
    def pixelwalk(self, sigma, walk_length=256):
        pixel = np.array((0, 0))
        plist = [pixel]
        bias = random.randint(-1, 1)
        while pixel[1] < walk_length:
            nx = -4.*np.arctan2(pixel[0], walk_length-pixel[1])/(np.pi)+4.5
            nx = np.log(nx/(8.-nx))
            nx = random.gauss(nx, sigma*random.betavariate(2, 5))
            nx = np.array(convert(bias + 8./(1.+np.exp(-nx))))
            plist.append(plist[-1]+nx)
            pixel += nx
        plist.pop(0)
        return plist
    def generatewalks(self, num_walks=10):
        '''Randomly generates walks up to num_walks'''
        self.walks = [self.pixelwalk(4.*np.random.rand(), 128) for _ in range(num_walks)]
    def randomwalk(self, default_freq=0.1, verbose=False, min_length=10):
        """Start a random walk at a random point along the edge of the leaf until it intersects with the leaf again.
                default_freq: Frequency at which completely uneaten leaves should appear
                verbose: print debug statements
                min_length: minimum contour length required to be considered valid"""
        size = np.ceil(1./default_freq)
        i = 0
        if np.random.randint(0, size) == 0:
            if verbose: print "Returning self"
            return self, 1.
        while True:
            i+=1
            length = len(self.contour)
            t = np.random.randint(0, length/2)
            p0 = np.asarray(self.contour[-t]).astype(int)
            windex = np.random.randint(0, len(self.walks))
            walk = self.walks[windex]
            L = [p0 + _ for _ in walk]
            L.insert(0, p0)
            L = [tuple(_) for _ in L]
            f = lambda x: (x in self.contour) and (x != list(p0))
            g = lambda x: (x[0]<0) or (x[1]<0)

            out_of_bounds = map(g, L)
            if any(out_of_bounds):
                if verbose: print "Out of bounds"
                continue
            try:
                filtered = filter(f, L)
                index = self.contour.index(filtered[-1])
            except IndexError:
                if verbose: print "Walk does not intersect leaf"
                continue

            if index > length-t:
                # Short Walk
                windex = L.index(filtered[-1])
                new = self.contour[:-t]
                img = np.copy(self.image)
                for p in L[:windex]:
                    img[:p[0], p[1]] = 0.
                l = L[:windex]
                img[:l[-1][0], :l[-1][1]] = 0.
                img[:l[0][0], :l[0][1]] = 0.
                new.extend(L[:windex])
                new.extend(self.contour[index:])

            else:
                # Long Walk
                new = self.contour[:index]
                L.reverse()
                windex = L.index(filtered[-1])
                img = np.copy(self.image)
                for p in L[windex:]:
                    img[p[0]:, p[1]] = 0.
                l = L[windex:]
                img[l[-1][0]:, :l[-1][1]] = 0.
                img[l[0][0]:, l[0][1]:] = 0.
                new.extend(L[windex:])
                new.extend(self.contour[-t:])
            if len(new) < min_length:
                continue
            break
        weight = float(len(new))/float(length)
        return Leaf(img, contour=new, scale=self.scale, img_from_contour=True, sigma=self.sigma, orient=False, name=self.__name__, rescale=False), weight
    def randomdata(self, base_size=20, data_size=100, **kwargs):
        """Creates a list of randomly scaled and rotated partially eaten leaves.
                base_size: number of unique random walks to perform
                data_size: number of inputs to the network"""
        data = list()
        target = list()
        weights = list()
        leaves = list()
        weight = list()
        for i in range(base_size):
            leaf, w = self.randomwalk(verbose=True)
            leaves.append(leaf)
            weight.append(w)


        size = 0
        while size < data_size:
            scale = 1.5*np.random.rand()+0.5
            random.jumpahead(size)
            angle = 2.*np.pi*np.random.rand()-np.pi
            index = np.random.randint(0, len(leaves))
            leaf = leaves[index]
            w = weight[index]
            new_leaf = tf.rotate(leaf.image, angle*180./np.pi, resize=True, interp='bicubic')
            new_leaf = imresize(new_leaf, scale, interp='bicubic')
            new_leaf = new_leaf[:64, :64]
            leaf_c = tf.rotate(leaf.cimage, angle*180./np.pi, resize=True, interp='bicubic')
            leaf_c = imresize(leaf_c, scale, interp='bicubic')

            new_leaf = Leaf(new_leaf, contour=parametrize(leaf_c), scale=scale, sigma=self.sigma, orient=False, name='eaten leaf %r of %r' %(size, self.__name__), rescale=False)

            data.append(new_leaf)
            target.append((scale, angle))
            weights.append(w)
            size += 1
        return data, target, weights

def test(pickled):
    minticks = 10
    scaleunits = 'mm'
    minsegmentarea = 0.1
    itermagnification = 2
    debug = False
    width, height = 0, 0 #dimensions of grey_image
    if not pickled:
        # Read Images
        dir = os.path.dirname(__file__)
        ref_im_file = os.path.join(dir, "../images/TEST/reference.jpg")
        # query_im_file = os.path.join(dir, "../images/TEST/query.jpg")

        grey_ref = sp_imread(ref_im_file, flatten=True)
        color_ref = sp_imread(ref_im_file)

        # grey_query = sp_imread(query_im_file, flatten=True)
        # color_query = sp_imread(query_im_file)

        # grey_hpixels, grey_wpixels = grey_ref.shape
        x = [1, 2, 3]
        y = ['a', 'b', 'c']
        z = zip(x,y)
        print zip(*z)


        # Get image size properties
        if width==0 or height==0:
            grey_scale = get_scale(grey_ref, minticks, scaleunits)
        else:  # width and height in cm specified by user, don't need to calculate scale
            found = True
            # The scale should be the same calculated from the width or the height, but average the two,
            # just in case there is some small discrepency.
            grey_scale = (width/float(n) + height/float(m)) * 0.5

        print "Segmenting Ref"
        segmented_ref = segment(grey_ref, color_ref, ref_im_file, 50, itermagnification=2, debug=False, scale=grey_scale, minsegmentarea=0.1, datadir="./")

        ref_leaflets = []
        ref_names = []
        ref_scales = []
        ref_contours = []

        for i in range(segmented_ref[3]):
            # print i
            leaf, scale, contour = iso_leaf(segmented_ref[0], i+1, ref_image=color_ref)
            # print leaf.shape
            ref_leaflets.append(leaf)
            # ref_leaflets.append(image2gray(leaf))
            # print leaf.shape
            ref_scales.append(scale)
            ref_contours.append(contour)
            ref_names.append("leaflet %r" % i)

        ref_leaflets.pop(6)
        ref_scales.pop(6)
        ref_contours.pop(6)
        ref_names.pop(6)
        data = TrainingData(ref_leaflets, ref_contours, ref_scales, names=ref_names, sigma=0.075*64)

    print 'Generating Data'
    if not pickled:
        inputs, inputs2D, targets, weights = data.generatedata(ninputs=100)
        cPickle.dump(data, open("leafdump.p", "wb"))
        cPickle.dump(inputs, open("inputdump.p", "wb"))
        cPickle.dump(inputs2D, open("input2Ddump.p", "wb"))
        cPickle.dump(targets, open("targetdump.p", "wb"))
        cPickle.dump(weights, open("weightdump.p", "wb"))
    else:
        data        = cPickle.load(open("leafdump.p", "rb"))
        inputs      = cPickle.load(open("inputdump.p", "rb"))
        inputs2D    = cPickle.load(open("input2Ddump.p", "rb"))
        targets     = cPickle.load(open("targetdump.p", "rb"))
        weights     = cPickle.load(open("weightdump.p", "rb"))
    print "Sigma: ", data.sigma
    inputs = np.asarray(inputs)

    print
    targets = np.asarray(targets)
    weights = np.ones_like(weights)

    pickled = False
    if pickled:
        print "Net Loaded"
        LeafNet =cPickle.load(open("netdump.p", "rb"))
    else:
        print 'Building Network'
        LeafNet = LeafNetwork(inputs, inputs2D, targets)
        cPickle.dump(LeafNet, open("netdump.p", "wb"))
        print 'Network Constructed'
        pickled = True


    angles          = np.linspace(0, 2.*np.pi*(1.-1./float(ninputs)), ninputs)
    test_leaf, _    = data.leaves[0].randomwalk()
    # contour = parametrize(test_leaf.image)
    # shape = np.amax(contour)
    # shape = (shape+1, shape+1)
    # img = np.zeros(shape, bool)
    # for c in contour:
    #     img[c] = True
    img = imresize(test_leaf.image, 1.3, interp='bicubic')
    print "Before"
    plt.imshow(img)
    plt.show()
    img = img[:64, :64]
    print img.shape
    img = np.array(img)
    # im = np.zeros((64, 64), img.dtype)
    # for i in range(64):
    #     for j in range(64):
    #         im[i, j] = np.sum(img[i, j, :]/3.)
    #         for c in range(3):
    #             im[c, i, j] = img[i, j, c]
    test_leaf = Leaf(img, orient=False, sigma=0.1*256, name="test")
    im = list()
    for a in angles:
        c = test_leaf.curvatures[test_leaf.extractpoint(a)]
        if np.isnan(c):
            print "c is a nan"
            c = 1e8
        im.append(c)
    print len(im)
    print np.array(im, ndmin=1).shape
    # test_points = np.array([test_leaf.curvatures[test_leaf.extractpoint(a)] for a in angles], ndmin=2)
    #
    print test_leaf.image.shape
    result = LeafNet.solve(np.asarray(im))
    # result = LeafNet.network.activate(im)
    print result
    print "Solution to s=1.3, a=0"
    scale, angle = result[0][:]
    scale = 2.**scale
    angle = 2.*np.pi*(angle+0.5)
    print scale, angle
    c = parametrize(img)
    print "Scale error: ", (1.3-2.**result[0][0])/1.3
    print "Angle error: ", 1.-(2.*np.pi+result[0][1])/(2.*np.pi)
    img = tf.rotate(img, -angle*180./np.pi, resize=True)
    if scale != 0: img = imresize(img, 1./scale, interp='bicubic')
    # print np.ptp(contour, 0)
    # print np.ptp(c, 0)
    print "After"
    plt.imshow(img)
    plt.show()

    test_leaf, _ = data.leaves[3].randomwalk()
    # contour = parametrize(test_leaf.image)
    # shape = np.amax(contour)
    # shape = (shape+1, shape+1)
    # img = np.zeros(shape, bool)
    # for c in contour:
    #     img[c] = True

    img = imresize(test_leaf.image, 0.7, interp='bicubic')
    img = tf.rotate(img, 45, resize=True)
    print "Before"
    plt.imshow(img)
    plt.show()
    img = img[:64, :64]
    print img.shape
    img = np.array(img)
    # im = np.zeros((64, 64), img.dtype)
    # for i in range(64):
    #     for j in range(64):
    #         im[i, j] = np.sum(img[i, j, :]/3.)
    #         for c in range(3):
    #             im[c, i, j] = img[i, j, c]
    test_leaf = Leaf(img, orient=False, sigma=0.1*256, name="test")
    im = list()
    for a in angles:
        c = test_leaf.curvatures[test_leaf.extractpoint(a)]
        if np.isnan(c):
            print "c is a nan"
            c = 1e8
        im.append(c)
    print len(im)
    print np.array(im, ndmin=2).shape
    # test_points = np.array([test_leaf.curvatures[test_leaf.extractpoint(a)] for a in angles], ndmin=2)
    #
    print test_leaf.image.shape
    result = LeafNet.solve(np.array(im, ndmin=2))
    # result = LeafNet.network.activate(im)
    print result
    print "Solution to s=0.7, a=45"
    scale, angle = result[0][:]
    scale = 2.**scale
    angle = 2.*np.pi*(angle+0.5)
    print scale, angle
    c = parametrize(img)
    print "Scale error: ", (.7-2.**result[0][0])/.7
    print "Angle error: ", (np.pi/4.-2.*np.pi*(result[0][1]-0.5))/(np.pi/4.)
    img = tf.rotate(img, -angle*180./np.pi, resize=True)
    if scale != 0: img = imresize(img, 1./scale, interp='bicubic')
    # print np.ptp(contour, 0)
    # print np.ptp(c, 0)
    print "After"
    plt.imshow(img)
    plt.show()

if __name__=='__main__':
    # pickled = False
    pickled = True
    # while True:
    if pickled:
        data = cPickle.load(open("leafdump.p", "rb"))
        inputs = cPickle.load(open("inputdump.p", "rb"))
        targets = cPickle.load(open("targetdump.p", "rb"))
        weights = cPickle.load(open("weightdump.p", "rb"))
        # LeafNet = cPickle.load(open("netdump.p", "wb"))

    pickled = test(pickled)