import numpy as np
import mahotas as mh
import numpy.linalg as la
from scipy.ndimage.filters import sobel, gaussian_filter, gaussian_filter1d
from scipy.misc import imresize
from scipy.ndimage import imread as sp_imread
from scipy.ndimage.interpolation import shift, zoom, rotate
from skimage import transform as tf

# Kernel for erosion and dilation
se = np.ones((3,3))

def curvature(contour, sigma=1., **kwargs):
    """
    Computes the curvature at each point along the edge of a contour
    :param contour:     list of tuples,
                        list of (x,y) coordinates defining a contour
    :param sigma:       float, optional
                        Width of the gaussian kernel to be convolved along the contour.
    :return: k:         list of floats,
                        list of curvature values along the contour
    """
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
    """

    :param leafimage:
    :param minticks:
    :param scaleunits:
    :return:
    """
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

def rgb_to_grey(image):
    assert len(image.shape) is 3, "Image is not RGB. Expected image with three dimensions, got %s instead" %(image.shape)
    color_channel = np.argmin(image.shape)
    shape = list()
    for i in range(len(image.shape)):
        if i != color_channel:
            shape.append(image.shape[i])

    grey_image = np.zeros(shape, dtype=float)

    if color_channel == 0:
        for i in range(shape[0]):
            for j in range(shape[1]):
                grey_image[i, j] = np.mean(image[:, i, j])
    elif color_channel == 1:
        for i in range(shape[0]):
            for j in range(shape[1]):
                grey_image[i, j] = np.mean(image[i, :, j])
    elif color_channel == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                grey_image[i, j] = np.mean(image[i, j, :])
    return grey_image


def gradient_orientation(image):
    """
    Calculate the gradient orientation for edge point in the image
    :param image:   ndimage,
                image for which the gradient orientation is to be calculated
    :return: gradient:  array,
                an array holding the gradient orientation in radians for each point in the image
    """
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    gradient = np.arctan2(dy, dx)
    return gradient


def normalize(image, weight=0.5):
    """
    Normalizes an image by setting pixels which fall below a certain threshhold to 0
    :param image: ndimage,
                Image to be normalized
    :param weight: float, optional
                A factor specifying the fraction of the range to use as the threshhold. A value of 0.5 denotes that any
                pixel that is less than halfway between the max and min values will be set to 0.
    :return: image: ndimage,
                normalized image with integer dtype
    """
    minvalue = np.min(image)
    maxvalue = np.max(image)
    thresh = weight * (maxvalue - minvalue) + minvalue

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= thresh:
                image[i, j] = 1
            else:
                image[i, j] = 0
    return image.astype(int)


def outline(image, alignment='x', indexing='coord'):
    """
    Creates an outline of an image by scanning along an axis and stoping at the first pixel which is not 0
    :param image: ndimage,
                image to be scanned
    :param alignment: string, optional
                which axis to scan across. Accepts 'x' or 'y'. Specifying 'x' will scan along the first dimension (rows)
                of the image, 'y' will scan across the second dimension (columns)
    :param indexing: string, optional
                Whether to index pixel locations by their (y,x) coordinates or their flattened array indeces. Specifying
                'coord' will use (y,x) coordinates, 'ravel' will used flattened array indeces.
    :return: pixels: list of edge pixel locations, see 'indexing' for possible output types.
    """
    pixels = [[], []]
    image = normalize(image)
    if alignment == 'x':
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] != 0:
                    pixels[1].append(i)
                    pixels[0].append(j)
                    break
            for j in range(image.shape[1]):
                if image[i, -j] != 0:
                    pixels[1].append(i)
                    pixels[0].append(image.shape[1] - j)
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
                    pixels[1].append(image.shape[0] - i)
                    pixels[0].append(j)
                    break
    plt.scatter(pixels[0][:], pixels[1][:])
    plt.show()
    if indexing == 'coord':
        return pixels
    elif indexing == 'ravel':
        pixels = np.ravel_multi_index(pixels, image.shape)
        return pixels

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