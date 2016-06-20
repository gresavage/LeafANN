# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:18:10 2015

@author: Tom
Based on code from Vinnie Monaco: https://github.com/vmonaco/general-hough
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
from progressbar import ProgressBar, Bar, Percentage
from collections import defaultdict
from scipy.ndimage import imread as sp_imread
from scipy.misc import imresize, imrotate
from skimage import transform as tf
from scipy.ndimage.filters import sobel, gaussian_filter, gaussian_filter1d
from leafarea import *
from image_processing import *


class TrainingData:
    def __init__(self, leaves, contours, scales, sigma=1., img_from_contour=False, names=None):
        print 'Training Data'
        self.sigma = sigma
        if names is None:
            names = self.namegen(names)
        self.leaves = [Leaf(leaves[_],
                            contour=contours[_],
                            scale=scales[_],
                            sigma=self.sigma,
                            img_from_contour=img_from_contour,
                            name=names[_],
                            rescale=True) for _ in range(len(leaves))]
        self.contours = [leaf.contour for leaf in self.leaves]
        self.trainingdata = list()
        self.trainingdata2D = list()
        self.trainingtargets = list()
        self.trainingweights = list()
        self.update()
        print "Initial Leaves Added"
        self.eaten_leaves = list()

    def namegen(self):
        i = 0
        print 'namegen'
        while True:
            yield "training leaf %r " % i
            i += 1
            if i == len(leaves):
                break

    @property
    def __len__(self):
        self.len = len(self.leaves)
        return self.len

    def update(self, sigma=None, visualize=False):
        """Bookkeeping method to make sure all data is consistent"""
        if sigma is not None:
            self.sigma = sigma
        self.smoothcontour(self.sigma)
        self.orient(visualize=visualize)
        self.findcentroid()
        self.getangles(sigma=self.sigma)
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
        angles = np.linspace(0., 2. * np.pi * (1. - 1. / float(ninputs)), ninputs)
        for leaf in self.leaves:
            leaf.generatewalks(20)
            data, targets, weights = self.randomdata(leaf, **kwargs)
            if verbose:
                print "Data length:"
                print len(data)
            for d in data:
                cdata = list()
                d.curvature(self.sigma)
                if verbose:
                    print "Image Shape: ", d.color_image.shape
                for a in angles:
                    try:
                        c = d.curvatures[d.extractpoint(a)]
                        if np.isnan(c):
                            if verbose:
                                print "C IS STILL A NAN"
                                print
                            c = 1E9
                        cdata.append(c)
                    except ValueError:
                        print "Curvature Data ValueError"
                        print "Warning: Curvature not defined for angle %r in leaf %r" % (180. * a / np.pi, d.__name__)
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
                if verbose:
                    print "New Shape: ", dimage.shape
                    print
                self.trainingdata2D.append(dimage)
            self.trainingtargets.extend(targets)
            self.trainingweights.extend(weights)
        self.trainingdata2D = np.concatenate(self.trainingdata2D, axis=0)
        if verbose:
            print self.trainingdata2D.shape
        return self.trainingdata, self.trainingdata2D, self.trainingtargets, self.trainingweights

    def randomdata(self, leaf, data_size=100, square_length=64, verbose=False, **kwargs):
        """Creates a list of randomly scaled and rotated partially eaten leaves"""

        if verbose:
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
            scale = 1.5 * np.random.rand() + 0.5
            angle = 2. * np.pi * np.random.rand() - np.pi
            index = np.random.randint(0, len(leaf_list))
            leaf = leaf_list[index]

            if not leaf.image.any():
                if verbose:
                    print "None, ever"
                plt.imshow(leaf.image)
                plt.show()
                plt.imshow(leaf.image.astype(bool))
                plt.show()
                continue
            new_leaf = tf.rotate(leaf.image, angle * 180. / np.pi, resize=True)
            if not new_leaf.any():
                if verbose:
                    print "None after TF"
                plt.imshow(new_leaf)
                plt.show()
                continue
            new_leaf = imresize(new_leaf, scale, interp='bicubic')
            if not new_leaf.any():
                if verbose:
                    print "None after RZ"
                plt.imshow(new_leaf)
                plt.show()
                continue

            leaf_c = tf.rotate(leaf.cimage, angle * 180. / np.pi, resize=True)
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
                print angle * 180. / np.pi
                print scale
                img = np.zeros(np.amax(leaf_c, 0) + 1, bool)
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
                new_leaf = Leaf(new_leaf, contour=leaf_c, scale=scale, sigma=leaf.sigma, orient=False, rescale=False,
                                name='eaten leaf %s of %s' % (size, leaf.__name__))
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
                        self.color_image = image / 256.
                    elif np.issubdtype(np.max(image), float):
                        print "NumPy subdtype float"
                        print np.issubdtype(np.max(image, float))
                        self.color_image = image
                    image = rgb_to_grey(image)
                else:
                    self.color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
                    self.color_image[:, :, 0] = image / 256.
                    self.color_image[:, :, 1] = image / 256.
                    self.color_image[:, :, 2] = image / 256.
                img = image.astype(bool)
                plt.imshow(img.astype(bool))
                plt.show()
                super(Leaf, self).__init__(parametrize(img), sigma=sigma, rescale=rescale, name=self.__name__)
                self.image = image
        else:
            if len(image.shape) == 3:
                self.color_image = image / 256.
                image = rgb_to_grey(image)
            else:
                self.color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
                self.color_image[:, :, 0] = image / 256.
                self.color_image[:, :, 1] = image / 256.
                self.color_image[:, :, 2] = image / 256.
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
        return np.sqrt(float(p0[0] - p1[0]) ** 2. + float(p0[1] - p1[1]) ** 2.)

    def pixelwalk(self, sigma, walk_length=256):
        pixel = np.array((0, 0))
        plist = [pixel]
        bias = random.randint(-1, 1)
        while pixel[1] < walk_length:
            nx = -4. * np.arctan2(pixel[0], walk_length - pixel[1]) / (np.pi) + 4.5
            nx = np.log(nx / (8. - nx))
            nx = random.gauss(nx, sigma * random.betavariate(2, 5))
            nx = np.array(convert(bias + 8. / (1. + np.exp(-nx))))
            plist.append(plist[-1] + nx)
            pixel += nx
        plist.pop(0)
        return plist

    def generatewalks(self, num_walks=10):
        '''Randomly generates walks up to num_walks'''
        self.walks = [self.pixelwalk(4. * np.random.rand(), 128) for _ in range(num_walks)]

    def randomwalk(self, default_freq=0.1, verbose=False, min_length=10):
        """Start a random walk at a random point along the edge of the leaf until it intersects with the leaf again.
                default_freq: Frequency at which completely uneaten leaves should appear
                verbose: print debug statements
                min_length: minimum contour length required to be considered valid"""
        size = np.ceil(1. / default_freq)
        i = 0
        if np.random.randint(0, size) == 0:
            if verbose:
                print "Returning self"
            return self, 1.
        while True:
            i += 1
            length = len(self.contour)
            t = np.random.randint(0, length / 2)
            p0 = np.asarray(self.contour[-t]).astype(int)
            windex = np.random.randint(0, len(self.walks))
            walk = self.walks[windex]
            L = [p0 + _ for _ in walk]
            L.insert(0, p0)
            L = [tuple(_) for _ in L]
            f = lambda x: (x in self.contour) and (x != list(p0))
            g = lambda x: (x[0] < 0) or (x[1] < 0)

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

            if index > length - t:
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
        weight = float(len(new)) / float(length)
        return Leaf(img, contour=new, scale=self.scale, img_from_contour=True, sigma=self.sigma, orient=False,
                    name=self.__name__, rescale=False), weight

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
            scale = 1.5 * np.random.rand() + 0.5
            random.jumpahead(size)
            angle = 2. * np.pi * np.random.rand() - np.pi
            index = np.random.randint(0, len(leaves))
            leaf = leaves[index]
            w = weight[index]
            new_leaf = tf.rotate(leaf.image, angle * 180. / np.pi, resize=True, interp='bicubic')
            new_leaf = imresize(new_leaf, scale, interp='bicubic')
            new_leaf = new_leaf[:64, :64]
            leaf_c = tf.rotate(leaf.cimage, angle * 180. / np.pi, resize=True, interp='bicubic')
            leaf_c = imresize(leaf_c, scale, interp='bicubic')

            new_leaf = Leaf(new_leaf, contour=parametrize(leaf_c), scale=scale, sigma=self.sigma, orient=False,
                            name='eaten leaf %r of %r' % (size, self.__name__), rescale=False)

            data.append(new_leaf)
            target.append((scale, angle))
            weights.append(w)
            size += 1
        return data, target, weights



def squaredim(image, shape=(100, 100)):
    """
    Squares an image to the specified shape
    :param image: ndimage
                Image to be reshaped
    :param shape: tuple of ints, optional
                Tuple specifying shape of the output image
    :return: squared_image, scaler
            squared_image: ndimage,
                        the image squared to the specified shape
            scaler: float,
                        the scaling factor used to fit the image to the specified shape
    """
    maxdim = np.max(image.shape)
    axis = image.shape.index(maxdim)
    scaler = 100./float(maxdim)
    squared_image = imresize(image, scaler, interp='nearest')
    plt.imshow(squared_image)
    plt.show()
    print squared_image.shape    
    print "-"*50
    return squared_image, scaler

def iso_leaf(segmented_image, leaf_num, square_length=64, ref_image=None, rescale=True):
    """
    Isolates a specific leaf in a labeled image
    :param segmented_image: ndimage
                            Image to be scanned for the leaf specified by leaf_num
    :param leaf_num:        int
                            Integer value of pixels which are the leaf of interese
    :param square_length:   int, optional
                            length of the side of the square image to be used for rescaling 
    :param ref_image:       ndimage, optional
                            Full color reference image to be used for a full-color analysis
    :param rescale:         Bool, optional
                            Whether to scale the image to the square length
    :return: leaflet, scale
                            leaflet: the image cropped to contain only the leaflet specified by leaf_num
                            scale:  the scale of the output image, 1. unless rescale was set to True
    """
    scale = 1.
    rows, cols = np.where(segmented_image == leaf_num)
    left = np.min(cols)
    top = np.min(rows)
    # Set the dimensions to be the max between the row and column ranges
    dimspan = max(np.ptp(zip(rows, cols), 0))

    if ref_image is not None:
        ref_image = ref_image.astype(float)
        leaflet = np.ones(ref_image.shape, float)
        for p in zip(rows, cols):
            leaflet[p] = ref_image[p]
        leaflet = leaflet[top:(top+dimspan+1), left:(left+dimspan+1)]
    else:
        leaflet = np.zeros(segmented_image.shape[:2], bool)
        i = 0
        for pixel in zip(rows, cols):
            leaflet[pixel] = True
            i+=1
        leaflet = leaflet[top:(top + dimspan + 1), left:(left + dimspan + 1)]
    scale = float(square_length)/float(dimspan)
    leaflet = imresize(leaflet, scale)
    contour = np.zeros(segmented_image.shape[:2], bool)
    for c in zip(rows, cols):
        contour[c] = True

    contour = contour[top-1:(top + dimspan + 1), left-1:(left + dimspan + 1)]
    contour = parametrize(contour)

    padding = [square_length - leaflet.shape[0]-1, square_length - leaflet.shape[1]-1]
    if padding[0] >= 0 or padding[1] >= 0:
        if ref_image is not None:
            padded = np.zeros((square_length, square_length, 3), float)
            shape = [s+1 for s in leaflet.shape[:2]]
            if shape[0] < 65 and shape[1] < 65:
                padded[1:shape[0], 1:shape[1]] = leaflet[:, :]
            elif shape[1] < 65:
                padded[1:, 1:shape[1]] = leaflet[:63, :]
            else:
                padded[1:, 1:] = leaflet[:63, :63]
            leaflet = padded
        else:
            padded = np.zeros((square_length, square_length), bool)
            shape = [s+1 for s in leaflet.shape[:2]]
            try:
                if shape[0] < 65 and shape[1] < 65:
                    padded[1:shape[0], 1:shape[1]] = leaflet[:, :]
                elif shape[1] < 65:
                    padded[1:, 1:shape[1]] = leaflet[:63, :]
                elif shape[0] < 65:
                    padded[1:shape[0], 1:] = leaflet[:, :63]
                else:
                    padded[1:, 1:] = leaflet[:63, :63]
            except TypeError:
                print "Petiole shape: ", shape
                raise
            leaflet = padded
    else:
        print "Negative Padding"
    return leaflet, scale, contour


def general_hough_closure(reference_image, **kwargs):
    """
    Generator function to create a closure with the reference image and origin
    at the center of the reference image
    :param reference_image: ndimage
                        image to be used as a reference to build the R-table
    :return: f:     callable,
                    funciton which fills the accumulator according to the generalized Hough transform algorithm with
                    reference_image as the reference, and the argument to f as the query image.
    """
    referencePoint = (reference_image.shape[0]/2, reference_image.shape[1]/2)
    r_table = build_r_table(reference_image, referencePoint)
    
    def f(query_image):
        return accumulate_gradients(r_table, query_image, **kwargs)
    return f

def build_r_table(image, origin, polar=True, cornering=False):
    """
    Build the R-table from the given shape image and a reference point on the
    reference image
    :param image: ndimage,
                Image to be used as a reference and to build the R-table
    :param origin: tuple,
                Location of reference point in Image
    :param polar: Bool, optional
                Whether to use (dx, dy) pixel offset, or angle/radius to create R-table
    :param cornering: Bool, optional
                Whether to use corners as reference points. If false, then all edge pixels are used
    :return:
    """
    # Create an edge image by eroding and then subtracting the foreground
    if len(image.shape) >= 3:
        image = rgb_to_grey(image)
    if image.dtype is not int and np.max(image) <= 1:
        image = image*255
        image = image.astype(int)
    elif image.dtype is not int and np.max(image) > 1:
        image = image.astype(int)

    edges = image-mh.morph.erode(image, se)
    edges = normalize(edges)
    # Calculate Gradient Orientation in Degrees
    gradient = gradient_orientation(image)
    
    # Create Empty R-Table Dictionary
    r_table = defaultdict(list)
    
    for (i,j), value in np.ndenumerate(corners if cornering else edges):
         if value != 0:
             '''
             If the pixel is an edge, append the x,y relationship between the
             pixel and the origin to the r-table dictionary whose key is the
             gradient direction of the edge
             The offset of all pixels with a particular gradient orientation
             are stored:
             gradient direction: [(pixel1_x, pixel1_y), (pixel2_x, pixel2_y),...]
             '''
             if polar:
                 r = np.sqrt(float(i-origin[0])**2.+float(j-origin[1])**2.)
                 alpha = np.arctan2(-float(i-origin[0]), float(origin[1]-j))
                 r_table[gradient[i,j]].append((r, alpha))
             else:
                 r_table[gradient[i,j]].append((origin[0]-i, origin[1]-j))
    return r_table

def accumulate_gradients(r_table, grey_image, scales=np.linspace(0.5, 1.5, 20), angles=np.linspace(-(0.9)*np.pi/2., (0.9)*np.pi/2., 20), polar=True, normalizing=True, cornering=False, gradient_weight=False, curve_weight=False, show_progress=True, verbose=True):

    """
    Fill the accumulator using the r_table to determine candidate object locations
    :param r_table: dictionary,
                A dictionary containing {gradient angle: offset} key-value pairs for determining candidate object
                locations from an edge point
    :param grey_image: ndimage,
                Image to be scanned for the object of interest
    :param angles:  list, optional
                Angle search space
    :param scales:  list, optional
                Scale search space
    :param polar:   Bool, optional
                Whether the values in r_table are (dy, dx) coordinate offsets or (radius, angle) offsets.
    :param normalizing: Bool, optional
                Whether to normalized the query image before searching
    :param cornering: Bool, optional
                Whether corners were used as reference points
    :param gradient_weight: Bool, optional
                Whether to update the accumulator by the gradient magnitude. If False, accumulator entries are
                incremented
    :param curve_weight:    Bool, optional
                Whether to update the accumulator by the curvature at edge points.
    :param show_progress:   Bool, optional
                Whether to display a progress bar
    :param verbose: Bool, optional
                Whether to print updates to the screen
    :return: accumulator, angles, scales
            accumulator:    4D array,
                            The 4D accumulator holding the vote totals for all possible candidate locations
            angles:         list,
                            The list of angles over which the possible locations were determined
            scales:         list,
                            The list of scales over which the possible locations were determined
    """
    if len(grey_image.shape) >= 3:
        grey_image = rgb_to_grey(grey_image)
    if grey_image.dtype is not int and np.max(grey_image <= 1.):
        grey_image = grey_image*255
        grey_image = grey_image.astype(int)
    elif grey_image.dtype is not int and np.max(grey_image >= 1.):
        grey_image = grey_image.astype(int)

    query_edges = grey_image-mh.morph.erode(grey_image, se)

    if normalizing:
        query_edges = normalize(query_edges)

    gradient = gradient_orientation(grey_image)
    if show_progress:
        scanning_pixels = 0
        for (i,j), value in np.ndenumerate(query_edges):
            if value != 0: scanning_pixels+=1
        if verbose:
            print "Edge Pixels: %r"%(scanning_pixels)

    accumulator = np.zeros((grey_image.shape[0], grey_image.shape[1], len(scales), len(angles)))
    pixels_scanned = 0
    '''
    Net inputs: i, j, theta, rho, i*cos(theta), i*sin(theta), j*cos(theta), j*sin(theta)
    '''
    if show_progress:
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=scanning_pixels).start()
    if verbose:
        print "Entries: ", accumulator.shape[0]*accumulator.shape[1]*accumulator.shape[2]*accumulator.shape[3]
    for (i,j), value in np.ndenumerate(query_edges):
        pixels_scanned += 1
        if verbose:
            if pixels_scanned == query_edges.shape[0]*query_edges.shape[1]:
                print "Image Scanned"
                break
            elif pixels_scanned % 10e2 == 0:
                print pixels_scanned
        if value != 0:
            '''
            If the pixel at (i,j) is an edge then find the pixels with the same
            gradient direction in the r-table.
            accum_i, accum_j is the origin of the reference image on the query
            image
            If the origin of the point is in the image then increment the pixel
            location in the accumulator
            '''
            for pixels in r_table[gradient[i,j]]:
                if polar:
                    xp = pixels[0]*np.cos(pixels[1])
                    yp = pixels[0]*np.sin(pixels[1])
                else:
                    xp = float(pixels[1])
                    yp = float(pixels[0])
                for a in range(len(angles)):
                    for s in range(len(scales)):
                        accum_j, accum_i = int(float(j)-(xp*np.cos(angles[a])+yp*np.sin(angles[a]))*scales[s]), int(float(i)+(xp*np.sin(angles[a])-yp*np.cos(angles[a]))*scales[s])
                        if (0 <= accum_i < accumulator.shape[0]) and (0 <= accum_j < accumulator.shape[1]):
                            if gradient_weight:
                                accumulator[accum_i, accum_j, s, a] += gradient[i, j]
                            elif curve_weight:
                                ang = np.arctan2(-i+query_cont.centroid[0], j-query_cont.centroid[1])
                                accumulator[accum_i, accum_j, s, a] += query_cont.curvatures[query_cont.extractpoint(ang)]
                            else:
                                accumulator[accum_i, accum_j, s, a] += 1 + gradient[i, j]
            if show_progress:
                pixels_scanned+=1
                pbar.update(pixels_scanned)
    if show_progress:
        pbar.finish()
    return accumulator, scales, angles

def test_general_hough(gh, reference_image, query_image, path, query_index=0):
    '''
    Uses a GH closure to detect shapes in an image and create nice output
    '''
    ref_im = np.copy(reference_image)
    query_im = np.copy(query_image)
    accumulator, angles, scales = gh(query_im)

    plt.gray()
    
    fig = plt.figure(figsize=(32, 44))
    fig.add_subplot(2,2,1)
    plt.title('Reference image')
    plt.imshow(ref_im)
    plt.scatter(ref_im.shape[0]/2, ref_im.shape[1]/2, marker='o', color='y')
    
    fig.add_subplot(2,2,2)
    plt.title('Query image')
    plt.imshow(query_im)
    
    fig.add_subplot(2,2,3)
    plt.title('Accumulator')
    i, j, k, l = np.unravel_index(accumulator.argmax(), accumulator.shape)
    plt.imshow(accumulator[:, :, k, l])

    fig.add_subplot(2,2,4)
    plt.title('Detection')

    # top 5 results in red
    m = n_max(accumulator, 5)
    x_points = [pt[1][1] for pt in m]
    y_points = [pt[1][0] for pt in m]
    rots = [pt[1][2] for pt in m]
    scalers = [pt[1][3] for pt in m]
    plt.scatter(x_points, y_points, marker='o')
    result = tf.rotate(ref_im, angles[k]*180./np.pi, resize=False)
    result = imresize(result, scales[l])
    plt.imshow(query_im, alpha=0.5)
    plt.imshow(result, alpha=0.5)

    # top result in yellow
    plt.scatter([j], [i], marker='o', color='y')
    print "Angle = ", angles[k]*180./np.pi
    print "Scale = ", scales[l]
    print
    plt.show()

    dir = os.path.dirname(__file__)
    d, f = os.path.split(path)[0], os.path.splitext(os.path.split(path)[1])[0]
    plt.savefig(os.path.join(d, f + '_output_%s_scale%s.png' %(query_index, scales[l])))
    return

    
def n_max(a, n):
    """
    Return the N max elements and indices in a
    :param a: ndarray
                array of values to be searched for n maximum elements
    :param n: int,
                number of maximum elements to return
    :return: n_max: list,
                list of largest n elements
    """
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]

def test():
    minticks = 10
    scaleunits = 'mm'
    minsegmentarea = 0.1
    itermagnification = 2
    debug = False
    width, height = 0, 0 #dimensions of grey_image
    
    # Read Images
    dir = os.path.dirname(__file__)
    ref_im_file = os.path.join(dir, "../images/TEST/reference.jpg")
    query_im_file = os.path.join(dir, "../images/TEST/query.jpg")
    
    grey_ref = sp_imread(ref_im_file, flatten=True)
    scale, found, edge = metricscale(grey_ref, 10, 'cm')
    color_ref = sp_imread(ref_im_file)
    

    grey_query = sp_imread(query_im_file, flatten=True)
    color_query = sp_imread(query_im_file)
    
    grey_hpixels, grey_wpixels = grey_ref.shape

    # Get image size properties
    if width==0 or height==0:
        grey_scale = get_scale(grey_ref, minticks, scaleunits)
    else:  # width and height in cm specified by user, don't need to calculate scale
        found = True
        # The scale should be the same calculated from the width or the height, but average the two,
        # just in case there is some small discrepency.
        grey_scale = (width/float(n) + height/float(m)) * 0.5
    
    segmented_ref = segment(sp_imread(ref_im_file, flatten=True), sp_imread(ref_im_file), ref_im_file, 50, itermagnification=2, debug=False, scale=grey_scale, minsegmentarea=0.1, datadir="./")
    segmented_query = segment(sp_imread(query_im_file, flatten=True), sp_imread(query_im_file), query_im_file, 50, itermagnification=2, debug=False, scale=grey_scale, minsegmentarea=0.1, datadir="./")

    plt.imshow(segmented_ref[0])    
    plt.show()
    plt.imshow(segmented_query[0])
    plt.show()
    
    query_leaflets = []
    ref_leaflets = []
    # Isolate the individual leaflets in the ref and the query images
    for i in range(segmented_ref[3]):
        # ref_leaflets.append(imresize(iso_leaf(segmented_ref[0], i+1), 7, interp='nearest'))
        ref_leaflets.append(iso_leaf(segmented_ref[0], i+1))
        print "Ref_leaflets %s shape" %i
        print ref_leaflets[-1].shape
        print
    for i in range(segmented_query[3]):
        # query_leaflets.append(imresize(iso_leaf(segmented_query[0], i+1), 7, interp='nearest'))
        query_leaflets.append(iso_leaf(segmented_query[0], i+1))
        leaf_accumulator = general_hough_closure(ref_leaflets[0]) #gh function
        test_general_hough(leaf_accumulator, ref_leaflets[0], query_leaflets[i], query_im_file, query_index=i)

    
if __name__ == '__main__':
    test()
