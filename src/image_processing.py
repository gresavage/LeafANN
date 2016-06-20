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

class Contour(object):
    def __init__(self, contour, sigma=1., scale=1., rescale=False, **kwargs):
        self.contour = contour
        self.rows, self._cols = zip(*contour)
        self.sigma = sigma
        self.scale = scale
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
        self.findcentroid()
        self.getradii(smoothed=smoothed, sigma=sigma)
        if smoothed:
            length = len(self._smooth_rows)
            ts = np.arcsin([float(self._centroid[0] - self._smooth_rows[_]) / self._radii[_] for _ in range(length)])
            tc = np.arccos([float(self._smooth_cols[_] - self._centroid[1]) / self._radii[_] for _ in range(length)])
        else:
            length = len(self.contour)
            ts = np.arcsin([float(self._centroid[0] - self._rows[_]) / self._radii[_] for _ in range(length)])
            tc = np.arccos([float(self._cols[_] - self._centroid[1]) / self._radii[_] for _ in range(length)])
        thetas = []
        for j in range(length):
            if ts[j] < 0 and tc[j] > np.pi / 2.:
                thetas.append(2. * np.pi - tc[j])
            elif ts[j] < 0:
                thetas.append(2. * np.pi + ts[j])
            else:
                thetas.append(tc[j])
        self._angles = thetas
        return self._angles

    @property
    def radii(self):
        return self._radii

    @radii.setter
    def radii(self, radii):
        self._radii = list(radii)

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

        x = [float(x) for x in self._cols]
        y = [float(y) for y in self._rows]

        xu = gaussian_filter1d(x, self._sigma, order=1, mode='wrap')
        yu = gaussian_filter1d(y, self._sigma, order=1, mode='wrap')

        xuu = gaussian_filter1d(xu, self._sigma, order=1, mode='wrap')
        yuu = gaussian_filter1d(yu, self._sigma, order=1, mode='wrap')

        k = [(xu[i] * yuu[i] - yu[i] * xuu[i]) / np.power(xu[i] ** 2. + yu[i] ** 2., 1.5) for i in range(len(xu))]
        self._curvatures = k

        if kwargs.get('visualize', False):
            plt.plot(k)
            plt.show()
        return k

    def extractpoint(self, theta, **kwargs):
        '''Finds the index of the point with an angle closest to theta'''
        self.getangles(**kwargs)
        diff = list(np.abs(self.angles[:] - theta))
        return diff.index(np.min(diff))

    def findcentroid(self):
        self._centroid = (int(np.mean(self._rows)), int(np.mean(self._cols)))
        return self._centroid

    def getradii(self, smoothed=False, sigma=1.):
        self.smooth(sigma)
        if smoothed:
            self._radii = [np.sqrt((self._smooth_rows[_] - float(self._centroid[0])) ** 2. + (
                self._smooth_cols[_] - float(self._centroid[1])) ** 2.) for _ in range(self._smooth_len)]
        else:
            self._radii = [np.sqrt(float(
                (self._rows[_] - float(self._centroid[0])) ** 2. + (self._cols[_] - float(self._centroid[1])) ** 2.))
                           for _ in range(len(self._contour))]
        return self._radii

    def orient(self, smoothed=True, sigma=1., **kwargs):
        visualize = kwargs.get('visualize', False)
        rescale = kwargs.get('rescale', True)
        resize = kwargs.get('resize', True)
        preserve_range = kwargs.get('preserve_range', True)
        if visualize:
            print "Before Orientation"
            cols = int(1.1 * np.max(self._cols))
            rows = int(1.1 * np.max(self._rows))
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
        angles = angles - angle + np.pi / 2.
        self._angles = angles
        curve = [(self._radii[_] * np.sin(self._angles[_]), self._radii[_] * np.cos(self._angles[_])) for _ in
                 range(length)]
        dimspan = np.ptp(curve, 0)
        dimmax = np.amax(curve, 0)
        centroid = (dimmax[0] + dimspan[0] / 2, dimmax[1] + dimspan[1] / 2)
        curve = [(c[0] + centroid[0], c[1] + centroid[1]) for c in curve]

        self.cimage = tf.rotate(self.cimage, -angle * 180. / np.pi + 90., resize=True).astype(bool)
        self.cimage, scale, contour = iso_leaf(self.cimage, True)
        self.image = tf.rotate(self.image, -angle * 180. / np.pi + 90., resize=True)
        self.image = imresize(self.image, scale, interp="bicubic")
        self.scale *= scale
        curve1 = parametrize(self.cimage)
        length = len(curve)
        curve = [curve[(index + _) % length] for _ in range(length)]
        c1img = np.zeros_like(self.cimage)
        for c in curve1:
            c1img[c] = True
        self.cimage = c1img
        self.getangles(smoothed=smoothed, sigma=sigma)
        self.smooth(sigma=sigma)
        self.curvature(sigma=sigma)

        if visualize:
            print "After Orientation"
            shape = (2 * self.centroid[0], 2 * self.centroid[1])
            img = np.zeros(shape, bool)
            for c in self._contour:
                img[c] = True
            plt.imshow(img)
            plt.show()
            print '-' * 40
            print

    def plot(self, smoothed=False):
        if not smoothed:
            rowmax, colmax = np.max(self.rows), np.max(self.cols)
        else:
            rowmax, colmax = np.max(self.smooth_rows), np.max(self.smooth_cols)

        img = np.zeros((rowmax + 1, colmax + 1), bool)
        for c in self._contour:
            img[c] = True
        plt.imshow(img)
        plt.show()

    def smooth(self, sigma=1.):
        self._sigma = sigma
        self._smooth_rows = tuple(gaussian_filter1d(self._rows, self._sigma, order=0, mode='wrap'))
        self._smooth_cols = tuple(gaussian_filter1d(self._cols, self._sigma, order=0, mode='wrap'))
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

        img = np.zeros(ubounds[:] + 1, bool)

        if len(cont) > 0 or min(np.ptp(cont, 0)) > 1:
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