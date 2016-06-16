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

# Good for the b/w test images used
r_table_min_threshold = 0.75
r_table_max_threshold = 2
accum_min_threshold = 0.2
accum_max_threshold = 0.5
se = np.ones((3,3))


def curvature(contour, sigma=1., **kwargs):
    '''Computes the curvature at each point along the edge'''
    rows, cols = zip(*contour)
    x = [float(x) for x in cols]
    y = [float(y) for y in rows]

    xu = gaussian_filter1d(x, sigma, order=1, mode='wrap')
    yu = gaussian_filter1d(y, sigma, order=1, mode='wrap')

    xuu = gaussian_filter1d(xu, sigma, order=1, mode='wrap')
    yuu = gaussian_filter1d(yu, sigma, order=1, mode='wrap')

    k = [(xu[i] * yuu[i] - yu[i] * xuu[i]) / np.power(xu[i] ** 2. + yu[i] ** 2., 1.5) for i in range(len(xu))]
    if kwargs.get('visualize', False):
        plt.plot(k)
        plt.show()
    return k

def get_scale(leafimage, minticks, scaleunits):
    n, m = leafimage.shape
    
    scale, found, edge = metricscale(leafimage, minticks, scaleunits)
    if not found:   # try to find a scale after histogram equalization
        scale, found, edge = metricscale(leafimage, minticks, scaleunits, True, False)

    if not found:   # try to find a scale after Gaussian blur
        scale, found, edge = metricscale(leafimage, minticks, scaleunits, False, True)
    return scale

def squaredim(image, shape=(100, 100)):
    maxdim = np.max(image.shape)
    axis = image.shape.index(maxdim)
    scaler = 100./float(maxdim)
#    squared_image = np.zeros(shape)
    squared_image = imresize(image, scaler, interp='nearest')
       # squared_image = np.pad(squared_image, ((0, 100-squared_image.shape[0]), (0, 100-squared_image.shape[1])), 'minimum')
#    np.pad(squared_image, (100-squared_image.shape[0], 100-squared_image.shape[1]), 'edge')
    plt.imshow(squared_image)
    plt.show()
    print squared_image.shape    
    print "-"*50
    return squared_image, scaler

class Contour(object):
    def __init__(self, contour, sigma=1., scale=1., rescale=False, **kwargs):
        self.contour = contour
        self.rows, self._cols = zip(*contour)
        self.sigma = sigma
        self.scale = scale
        # self.findcentroid()
        self.cimage = self.contourtoimg(rescale=rescale).astype(bool)
        self.curvature(sigma=self._sigma)
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
        # self.findcentroid()
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
        self.centroid = (int(np.mean(self._rows)), int(np.mean(self._cols)))
        return self._centroid
    def getradii(self, smoothed=False, sigma=1.):
        self.smooth(sigma)
        if smoothed:
            self._radii = [np.sqrt((self._smooth_rows[_]-float(self._centroid[0]))**2.+(self._smooth_cols[_]-float(self._centroid[1]))**2.) for _ in range(self._smooth_len)]
        else:
            self._radii = [np.sqrt(float((self._rows[_]-float(self._centroid[0]))**2.+(self._cols[_]-float(self._centroid[1]))**2.)) for _ in range(len(self._contour))]
        return self._radii
    def orient(self, smoothed=True, sigma=1., **kwargs):
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
        angles = angles - angle + np.pi/2.
        self._angles = angles
        curve = [(self._radii[_]*np.sin(self._angles[_]), self._radii[_]*np.cos(self._angles[_])) for _ in range(length)]
        dimspan = np.ptp(curve, 0)
        dimmax = np.amax(curve, 0)
        centroid = (dimmax[0] + dimspan[0]/2, dimmax[1] + dimspan[1]/2)
        curve = [(c[0] + centroid[0], c[1] + centroid[1]) for c in curve]


        self.cimage = tf.rotate(self.cimage, -angle*180./np.pi+90., resize=True).astype(bool)
        self.cimage, scale, contour = iso_leaf(self.cimage, True)
        self.image = tf.rotate(self.image, -angle*180./np.pi+90., resize=True)
        self.image = imresize(self.image, scale, interp="bicubic")
        self.scale *= scale
        curve1 = parametrize(self.cimage)

        # Scale contour by scaling factor. Set 0 index to be top of leaflet
        length = len(curve)
        curve = [curve[(index+_)%length] for _ in range(length)]
        c1img = np.zeros_like(self.cimage)
        for c in curve1:
            c1img[c] = True
        self.cimage = c1img

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
        img, scale = iso_leaf(img, True, square_length=max(shape), rescale=rescale)
        cont = parametrize(img)
        img = np.zeros_like(img)
        for c in cont:
            img[c] = True
        if contour is None:
            self.cimage = img
            self.contour = cont
        return img.astype(bool)

def iso_leaf(segmented_image, leaf_num, square_length=64, ref_image=None, rescale=True):
    # Input:
    #   segmented_image: segmented image of leaf to be scanned
    #   leaf_num: number of leaf to be isolated from ref_image
    #   ref_image: full color reference image to be used
    # Output:
    #   petiole: image cropped to the dimensions of the petiole of interest
    # Finds the leaf specified by leaf_num in ref_image and returns an image
    # cropped to the smallest box capable of bounding said leaf.
    # print "iso_leaf"
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
    scale = float(square_length)/float(dimspan)
    petiole = imresize(petiole, scale)
    contour = np.zeros(segmented_image.shape[:2], bool)
    for c in zip(rows, cols):
        contour[c] = True
    contour = contour[top:(top + dimspan + 1), left:(left + dimspan + 1)]
    padding = [square_length - petiole.shape[0]-1, square_length - petiole.shape[1]-1]
    if padding[0] >= 0 or padding[1] >= 0:
        if ref_image is not None:
            padded = np.zeros((square_length, square_length, 3), float)
            shape = [s+1 for s in petiole.shape[:2]]
            if shape[0] < 65 and shape[1] < 65:
                padded[1:shape[0], 1:shape[1]] = petiole[:, :]
            elif shape[1] < 65:
                padded[1:, 1:shape[1]] = petiole[:63, :]
            else:
                padded[1:, 1:] = petiole[:63, :63]
            petiole = padded
        else:
            padded = np.zeros((square_length, square_length), bool)
            shape = [s+1 for s in petiole.shape[:2]]
            try:
                if shape[0] < 65 and shape[1] < 65:
                    padded[1:shape[0], 1:shape[1]] = petiole[:, :]
                elif shape[1] < 65:
                    padded[1:, 1:shape[1]] = petiole[:63, :]
                elif shape[0] < 65:
                    padded[1:shape[0], 1:] = petiole[:, :63]
                else:
                    padded[1:, 1:] = petiole[:63, :63]
            except TypeError:
                print "Petiole shape: ", shape
                raise
            petiole = padded
    else:
        print "Negative Padding"
    return petiole, scale


def general_hough_closure(reference_image, **kwargs):
    '''
    Generator function to create a closure with the reference image and origin
    at the center of the reference image
    
    Returns a function f, which takes a query image and returns the accumulator
    '''
    referencePoint = (reference_image.shape[0]/2, reference_image.shape[1]/2)
    r_table = build_r_table(reference_image, referencePoint)
    
    def f(query_image):
        return accumulate_gradients(r_table, query_image, **kwargs)
    return f

def scaleto(size, image):
    s = max(image.shape)
    i = image.shape.index[s]
    s *= float(size)/float(s)

def build_r_table(image, origin, polar=True, cornering=False):
    '''
    Build the R-table from the given shape image and a reference point on the
    reference image
    '''
    # Create an edge image by eroding and then subtracting the foreground
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
    
#    minvalue = np.min(edges)
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


def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    print "gradient orientation"
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    gradient = np.arctan2(dy, dx)
    return gradient

def normalize(image, weight = 0.5):
    minvalue = np.min(image)
    maxvalue = np.max(image)    
    thresh = weight*(maxvalue-minvalue)+minvalue
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] >= thresh:
                image[i,j] = 1
            else:
                image[i,j] = 0
    return image.astype(int)

def outline(image, alignment='x', indexing = 'coord'):
    # USE DEFAULTDICT TO AVOID DUPLICATES
    pixels = [[], []]
    image = normalize(image)
    if alignment == 'x':
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j] != 0:
                    pixels[1].append(i)
                    pixels[0].append(j)
                    break
            for j in range(image.shape[1]):
                if image[i,-j] != 0:
                    pixels[1].append(i)
                    pixels[0].append(image.shape[1]-j)
                    break
    if alignment == 'y':
        for j in range(image.shape[1]):
            for i in range(image.shape[0]):
                if image[i, j] != 0:
                    pixels[1].append(i)
                    pixels[0].append(j)
                    break
            for i in range(image.shape[0]):
                if image[-i, j] != 0:
                    pixels[1].append(image.shape[0]-i)
                    pixels[0].append(j)
                    break
    plt.scatter(pixels[0][:], pixels[1][:])
    plt.show()
    if indexing == 'coord':
        return pixels
    elif indexing == 'ravel':
        pixels = np.ravel_multi_index(pixels, image.shape)
        return pixels

def accumulate_gradients(r_table, grey_image, angles=np.linspace(-(0.9)*np.pi/2., (0.9)*np.pi/2., 20), scales=np.linspace(3./4., 1.25, 20), polar=True, normalizing=True, cornering=False, gradient_weight=False, curve_weight=False, show_progress=True, verbose=True):
    '''
    Perform a General Hough Transform with the given image and R-table
    grey_image is the query image
    '''
    print "accumulate_gradients"
    show_progress = False
    if len(grey_image.shape) >= 3:
        print "Grey image.shape"
        print grey_image.shape
        grey_image = rgb_to_grey(grey_image)
        print grey_image.shape
    if grey_image.dtype is not int and np.max(grey_image <= 1.):
        grey_image = grey_image*255
        grey_image = grey_image.astype(int)
    elif grey_image.dtype is not int and np.max(grey_image >= 1.):
        grey_image = grey_image.astype(int)

    query_edges = grey_image-mh.morph.erode(grey_image, se)

    if normalizing:
        query_edges = normalize(query_edges)
    print "using gradient_weight" if gradient_weight else "using unity weight"

    gradient = gradient_orientation(grey_image)
    # Display a progress bar
    if show_progress:
        scanning_pixels = 0
        for (i,j), value in np.ndenumerate(query_edges):
            if value != 0: scanning_pixels+=1
        print "Edge Pixels: ", scanning_pixels

    plt.imshow(query_edges)
    plt.show()

    accumulator = np.zeros((grey_image.shape[0], grey_image.shape[1], len(angles), len(scales)))
    pixels_scanned = 0
    '''
    Net inputs: i, j, theta, rho, i*cos(theta), i*sin(theta), j*cos(theta), j*sin(theta)
    '''
    if show_progress: pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=scanning_pixels).start()
    print "Entries: ", accumulator.shape[0]*accumulator.shape[1]*accumulator.shape[2]*accumulator.shape[3]
    for (i,j), value in np.ndenumerate(query_edges):
        pixels_scanned += 1
        if pixels_scanned == query_edges.shape[0]*query_edges.shape[1]:
            print "Image Scanned"
            break
            # raise StopIteration
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
                                accumulator[accum_i, accum_j, a, s] += gradient[i, j]
                            elif curve_weight:
                                ang = np.arctan2(-i+query_cont.centroid[0], j-query_cont.centroid[1])
                                accumulator[accum_i, accum_j, a, s] += query_cont.curvatures[query_cont.extractpoint(ang)]
                            else:
                                accumulator[accum_i, accum_j, a, s] += 1 + gradient[i, j]
                                # accumulator[accum_i, accum_j, a, s] += 1
            if show_progress:
                pixels_scanned+=1
                pbar.update(pixels_scanned)
    if show_progress: pbar.finish()
    return accumulator, angles, scales


def test_general_hough(gh, reference_image, query_image, path, query_index=0):
    '''
    Uses a GH closure to detect shapes in an image and create nice output
    '''
    # query_image, s = squaredim(query_image)
    ref_im = np.copy(reference_image)
    query_im = np.copy(query_image)
    accumulator, angles, scales = gh(query_im)
    # outline(query_image)

    # plt.clf()
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
    # print "5 Max Points"
    # print pt[1]
    # print
    rots = [pt[1][2] for pt in m]
    scalers = [pt[1][3] for pt in m]
    plt.scatter(x_points, y_points, marker='o')
    # print angles[rots]*180./np.pi
    # print scales[scalers]
    result = tf.rotate(ref_im, angles[k]*180./np.pi, resize=False)
    result = imresize(result, scales[l])
    plt.imshow(result)

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
    '''
    Return the N max elements and indices in a
    '''
    print "n_max"
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]

def test():
    print "test"
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

        ref_leaflets.append(imresize(iso_leaf(segmented_ref[0], i+1), 7, interp='nearest'))
    for i in range(segmented_query[3]):

        query_leaflets.append(imresize(iso_leaf(segmented_query[0], i+1), 7, interp='nearest'))

        leaf_accumulator = general_hough_closure(ref_leaflets[0]) #gh function
        test_general_hough(leaf_accumulator, ref_leaflets[0], query_leaflets[i], query_im_file, query_index=i)

    
if __name__ == '__main__':
    test()
   # square_test()
