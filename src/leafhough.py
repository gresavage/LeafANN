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
            # if pixels_scanned == np.dot(accumulator.shape):
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
