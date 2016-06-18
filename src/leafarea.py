# -*- coding: utf-8 -*-
# sys.path.append("C:\Users\hedonley\Documents\projects\Leaf Area\python code")
# from leafarea import *
# file = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos\\petioletest01.jpg' 
# file = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos\\petioletest02.jpg' 
# file = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos\\petioletest03.jpg' 
# file = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos\\petioletest04.jpg'
# file = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos test\\plot 1\\p1 date 1\\morrow-honeysuckle-test-01.JPEG'
# file = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos\\For Ed to Process\\JB INV End ps\\IMG_0277 inv09A.jpg'
# file = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos\\RePS Jan 15 2015\\JB INV End ps Jan 15 updated\\IMG_0277 inv09A.jpg'
# file = 'C:\\Users\\hedonley\\Documents\\projects\\Leaf Area\\leaf photos\\RePS Jan 22 2015\\JB INV End ps Jan 15 updated\\IMG_0261 inv04B.jpg'
# newimage, area, fillarea, numleaves, scale = leafarea(file)
__author__ = "Dr. H. Edward Donley"

import numpy as np
import os.path
#from mahotas import imread as 
import matplotlib
# Use a non-interactive backend for matplotlib, so that it does not display plots to the terminal.
#matplotlib.use('Agg')
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import mahotas as mh
from skimage import io
from skimage import exposure
from scipy import misc
from scipy import ndimage
#import Image
from PIL import Image, ImageFont, ImageDraw
import array

def rgb2gray(rgb):
    "Convert an RBG image to grayscale"
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def leafarea(file, iterationsscaled=20, itermagnification=2, debug=False, width=0, height=0, minsegmentarea=.1, minticks=10, scaleunits='mm', equalize=False, blur=False):
    "Find the areas of leaves in a photograph and the areas of the portions that are missing."
    # Input:
    #   file: pathname of file to be modified
    #   iterations: number of times to apply the 3x3 structuring element in erosion and dilation operations.
    #       More iterations are needed if the petiole is wide or if blemishes are large.
    #       Note: needed iterations = 60 for petioletest02.jpg
    #       Note: needed iterations = 50 for petioletest03.jpg
    #   width: width of image in cm (or inches)
    #   height: height of image in cm (or inches)
    #   minsegmentarea: minimum area (in cm^2 or inches^2) of each leaf in image.
    #       Any smaller objects are removed.
    #   scaleunits: units for tick marks along one edge of the image.
    #       Value can be 'mm', 'cm', 'inch', 'halfinch', 'quarterinch', 'eighthinch', 'sixteenthinch'
    #       Default is 'mm'
    #       The units for the area will be cm^2 or inch^2, depending on the value of scaleunits
    # Output:
    #   newlabeled: image with petioles and small background blemishes removed and segmented into individual
    #       leaves.  Each pixel is numbered according to the leaf in which it resides.  background = 0.
    #   leafarea: an array containing the areas of leaf 1 through numleaves.
    #   fillarea: an array containing the fill region of leaf 1 through numleaves.  This is the area eaten by herbivores.
    #   numleaves: the number of leaves in the image.
    #   scale: cm/pixel or inches/pixel, calculated from the ruler along the edge of the image.
#    from scikits.image import morphology, io

    # Write output image to same directory as the one containing the input image
    datadir = os.path.dirname(os.path.realpath(file))
    
    # Read the image and convert to gray scale.
    colorleafimage = mh.imread.imread(file)
    leafimage = rgb2gray(colorleafimage)
    n,m = leafimage.shape

    if debug:  # create a figure displaying intermediate images
        pylab.figure(1)
#        pylab.subplot(2,3,1)
        pylab.imshow(colorleafimage)        
  
    # If the iterationsscaled of erosions (used in the segment function) has not been 
    # specified by the user, then set it to be 50 for an image that is 2000x2000 pixels, 
    # but scaled according to the size of the image.
    iterations = int(iterationsscaled*max(n,m)/2000)

    if width == 0 or height == 0:  # width and height in cm not specified by the user
        # Find the ruler marks along one edge and calculate the scale (cm/pixel or inches/pixel)
        scale, found, edge = metricscale(leafimage, minticks, scaleunits)
#        print "width,height (cm) = ", n * scale, m * scale, "scale = ", scale

        if not found:   # try to find a scale after histogram equalization
            scale, found, edge = metricscale(leafimage, minticks, scaleunits, True, False)
#            print "width,height with histogram equalization (cm) = ", n * scale, m * scale, "scale = ", scale

        if not found:   # try to find a scale after Gaussian blur
            scale, found, edge = metricscale(leafimage, minticks, scaleunits, False, True)
#            print "width,height with histogram equalization (cm) = ", n * scale, m * scale, "scale = ", scale
    else:  # width and height in cm specified by user, don't need to calculate scale
        found = True
        # The scale should be the same calculated from the width or the height, but average the two,
        # just in case there is some small discrepency.
        scale = (width/float(n) + height/float(m)) * 0.5

    if found:
        newlabeled, leafarea, fillarea, numleaves = segment(leafimage, colorleafimage, file, iterations, itermagnification, debug, scale, minsegmentarea, datadir)
        return(newlabeled, leafarea, fillarea, numleaves, scale)
    else:
        # Could not find a scale along an edge of the photo
        return(0, 0, 0, 0, 0)

def metricscale(leafimage, minticks, scaleunits, equalize=False, blur=False):
    "Find the ruler marks along one edge of a binary image, in mm, and calculate the scale in cm per pixel."
    # Must have at least minticks tick marks along one edge
    # Input:
    #   file:     a string containing the path and file name for an image file
    #   minticks: must have at least this number of tick marks in the scale along one edge of the image
    #             in order to be considered to be reliable.  The default is 10.
    # Output:
    #   scale:    cm (or inches) per pixel in the image.  This is a floating point number.
    #   found:    whether or not at least minticks ruled marks were found along at least one edge.
    #             This is a boolean.
    #   edge:     edge along which ruled marks were found.  Either 'L', 'R', 'T', or 'B' for
    #             left, right, top, or bottom.

#    leafimage = imread.imread(file, as_grey=True)
#    leafimage = mh.demos.nuclear_image()

#    print 'before equalization, min = ', np.amin(leafimage), ', max = ', np.amax(leafimage)
    if equalize:
        leafimage = exposure.equalize_hist(leafimage/255.0)*255.0
#    pylab.imshow(leafimage, cmap=cm.gray)
#    print 'after equalization, min = ', np.amin(leafimage), ', max = ', np.amax(leafimage)
    if blur:
        leafimage = mh.gaussian_filter(leafimage, 4)
    threshold = mh.otsu(leafimage.astype(np.uint8))
#    print 'threshold = ', threshold
    leafimagebw = (leafimage > threshold)  # convert to BW image
#    pylab.imshow(leafimagebw, cmap=cm.gray, vmin=0, vmax=1)
        
    # Find the edge with the scale
    n, m = leafimage.shape
    leftright = [5, m-5]  # five pixels in from left and right edges, to avoid noise at image edges
    topbottom = [5, n-5]  # five pixels in from top and bottom edges
    found = False
    edge = ''

    # Check left and right edges for ruler marks    
    for i in topbottom:
        up = array.array('L', [])  # pixel locations where leafimage values go from 0 to 1.
        down = array.array('L', [])  # pixel locations where leafimage values go from 1 to 0.
        for j in range(1,m-1):
            if leafimagebw[i,j] < leafimagebw[i,j+1]:
                up.append(j)
            elif leafimagebw[i,j] > leafimagebw[i,j+1]:
                down.append(j)
        # The float variable, scale, is the number of pixels per cm.
        if i == 5:
            foundtop, scale = foundscale(up, down, n, m, minticks, scaleunits)  # Are ruler marks on top edge?
            if foundtop:  # no need to search any of the other edges for ruler marks.
                found = True
                edge = 'T'
                break
        else:
            foundbottom, scale = foundscale(up, down, n, m, minticks, scaleunits)  # Are ruler marks on bottom edge?
            if foundbottom:
                found = True
                edge = 'B'

    # If the ruler marks are not on the top or the bottom edges, check the left and the right    
    if not found:
        downcounter = 0  # number of times that leafimage value goes from 1 to 0.
        upcounter = 0  # number of times that leafimage value goes from 0 to 1.
        side = -1  # is this the right side or left side?
        for j in leftright:
            up = array.array('L', [])  # pixel locations where leafimage values go from 0 to 1.
            down = array.array('L', [])  # pixel locations where leafimage values go from 1 to 0.
            for i in range(1,n-1):
                if leafimagebw[i,j] < leafimagebw[i+1,j]:
                    up.append(i)
                elif leafimagebw[i,j] > leafimagebw[i+1,j]:
                    down.append(i)
            # The float variable, scale, is the number of pixels per cm.
            if j == 5:
                foundleft, scale = foundscale(up, down, n, m, minticks, scaleunits)  # Are ruler marks on left edge?
                if foundleft:  # no need to search any of the other edges for ruler marks.
                    found = True
                    edge = 'L'
                    break
            else:
                foundright, scale = foundscale(up, down, n, m, minticks, scaleunits)  # Are ruler marks on right edge?
                if foundright:
                    found = True
                    edge = 'R'

#    pylab.imshow(leafimage)
    
    return(scale, found, edge)
    
def foundscale(up, down, n, m, minticks, scaleunits):
    "Check up and down arrays for evenly spaced intervals indicating ruler marks."
    # Input:
    #   up: integer array of pixel locations where the image changes from 0 to 1.
    #   down: integer array of pixel locations where the image changes from 1 to 0.
    #   n: number of rows of pixels in the image.
    #   m: number of columns of pixels in the image.
    #   minticks: must have at least this number of tick marks in the scale along one edge of the image
    #             in order to be considered to be reliable.  The default is 10.
    # Output:
    #   found: logical indicating whether or not a ruler was found along this edge.
    #   scale: cm (or inches) per pixel, as a float.  Meaningful only if found is True.

    epsilon = 8000/max(n, m)  # allow tick mark widths to vary by up to + or - epsilon pixels        

    # Conversion factor from units of tick marks to units of image (for example, mm tick marks to cm for image)
    conversion = {'mm': 0.1, 'cm': 1.0, 'inch': 1.0, 'halfinch': 0.5, 'quarterinch': 0.25, 'eighthinch': 0.125, 'sixteenthinch': 0.0625}


    if len(up) < minticks:  # not enough ticks.
        return(False, 0)
    diff = array.array('L', [0]*(len(up)-1))
    for i in range(1, len(up)):
        diff[i-1] = up[i] - up[i-1]

#    print "diff = ", diff
                        
    diffsort = sorted(diff)
    difftruncated = diffsort[int(len(diff)/4):int(3*len(diff)/4)]

    # Create histogram of bin width, epsilon, to find most common distance between ruler marks.  Include only those in the interquartile range.
    diffmax = max(difftruncated)
    diffmin = min(difftruncated)
    diffbin = array.array('L', [0]*((diffmax-diffmin)/epsilon+1) )
    for i in range(len(difftruncated)):
        diffbin[int((difftruncated[i]-diffmin)/epsilon)] += 1

    # Find the mode of the histogram    
    binmax = diffbin[0]
    binmode = 0
    for i in range(1,len(diffbin)):
        if diffbin[i] > binmax:
            binmax = diffbin[i]
            binmode = i

    # Add the two adjacent bins to the largest bin
    binwidth = float(diffmax-diffmin)/float(epsilon)
    if binmode > 0:
        binmax += diffbin[binmode-1]
    if binmode < len(diffbin)-1:
        binmax += diffbin[binmode+1]
       
    # Find the second to the highest peak of the histogram, at least two bins from the mode
    bin2max = 0
    for i in range(len(diffbin)):
        if diffbin[i] > bin2max:
            if binmode - 1 <= i and i <= binmode + 1: # skip the mode and its immediate neighbors
                continue
            bin2max = diffbin[i]

    # Is the mode is more than 5 times as strong as the next to the highest mode?
    found = (binmax >= minticks and binmax/5 > bin2max)
    if not found:
        return(False, 0)

    # Find the typical ruler spacing
    if len(diffbin) <= 2:
        tickspacing = diffmin + float(binmode+0.5)*binwidth
    elif binmode == 0:
        tickspacing = diffmin + float((binmode+0.5)*binmax + (binmode+1.5)*diffbin[binmode+1])/float(diffbin[binmode]+diffbin[binmode+1])  *binwidth
    elif binmode == len(diffbin) - 1:
        tickspacing = diffmin + float((binmode-0.5)*diffbin[binmode-1] + (binmode+0.5)*binmax)/float(diffbin[binmode-1]+diffbin[binmode])  *binwidth
    else:
        tickspacing = diffmin + float((binmode-0.5)*diffbin[binmode-1] + (binmode+0.5)*binmax + (binmode+1.5)*diffbin[binmode+1])/float(diffbin[binmode-1]+diffbin[binmode]+diffbin[binmode+1])  *binwidth
        
    i = 0
    maxrunstart = 0
    maxrunend = 0
    while i < len(diff):
        if abs(diff[i] - tickspacing) < epsilon:
            runstart = i
            while i < len(diff) and abs(diff[i] - tickspacing) < epsilon:
                i += 1
            runend = i - 1
            if runend - runstart > maxrunend - maxrunstart:
                maxrunstart = runstart
                maxrunend = runend
        i += 1

    if maxrunend - maxrunstart < minticks:
        return(False, 0)
    else:
        # scale is in cm (or inches) per pixel
        scale = float(maxrunend-maxrunstart-1) / float(up[maxrunend-1] - up[maxrunstart]) * conversion[scaleunits]
        return(True, scale)

def segment(grayleafimage, colorleafimage, file, iterations, itermagnification, debug, scale, minsegmentarea, datadir):
    "Remove petioles and small blemishes from the background, fill holes, and segment the image into individual leaves."
    # Input:
    #   grayleafimage: grayscale image to be modified
    #   colorleafimage: orginal RGB image, used to create the feedback image
    #   file: filename of original image, from which the filename of the feedback image is derived
    #   iterations: number of times to apply the 3x3 structuring element in erosion and dilation operations.
    #       More iterations are needed if the petiole is wide or if blemishes are large.
    #       Note: needed iterations = 60 for petioletest02.jpg
    #       Note: needed iterations = 50 for petioletest03.jpg
    #   scale: cm (or inches) per pixel
    #   minsegmentarea: minimum area (in cm^2 or inches^2) of each leaf in image.
    #       Any smaller objects are removed.
    # Output:
    #   newlabeled: image with petioles and small background blemishes removed and segmented into individual
    #       leaves.  Each pixel is numbered according to the leaf in which it resides.  background = 0.
    #   leafarea: an array containing the areas of leaf 1 through numleaves.
    #   numleaves: the number of leaves in the image.
    
    # Convert minimum segment area to minimum number of pixels.
    m, n = grayleafimage.shape  # number of rows and columns in image
    minsegmentpixels = minsegmentarea / (scale*scale)

#    leafimage = (matplotlib.colors.rgb_to_hsv(grayleafimage)[:,:,2]).astype(np.uint8)
#    leafimage = leafimage[:,:,0]
    # Blur the image using a Gaussian blur with standard deviation, 4 pixels, to eliminate small blemishes
#    leafimage = mh.gaussian_filter(leafimage, 4)

    # Separate the leaves from the background using Otsu's method
    # Note: Do not use a Gaussian blur.  If a Gaussian blur is applied first to eliminate small blemishes, 
    # the holes in the leaves will be distorted.
    tleaf = mh.otsu(grayleafimage.astype(np.uint8)); # Find a gap in the histogram.
    leafimage = (grayleafimage < tleaf) # assign all pixels below this threshold to 1 and above threshold to 0
#    leafimage = (leafimage < leafimage.mean())  # If Otsu's method is not satisfactory, try using the mean as the threshold

#    if debug:
#        pylab.subplot(2,3,2)
#        pylab.imshow(leafimage, cmap=cm.gray, vmin=0, vmax=1)

    # Erode the image to eliminate thin petioles and small blemishes.  
    se = np.ones((3,3))
    newleafimage = leafimage
    for i in range(iterations):
        newleafimage = mh.morph.erode(newleafimage, se)
#    if debug:
#        pylab.subplot(2,3,3)
#       pylab.imshow(newleafimage, cmap=cm.gray, vmin=0, vmax=1)
    # Dilate to recover the full blade.  Overdilate, to recover skinny points on leaves
    for i in range(int(itermagnification*iterations)):
        newleafimage = mh.morph.dilate(newleafimage, se)

    # Take the intersection with original thresholded image to eliminate any overdilation.
    leafimage = np.multiply(newleafimage, leafimage)
  
    # Segment the image into individual leaves
    labeled, numleaves = mh.label(leafimage)
    
    leafarea = [] # area of each leaf in cm^2
    centroidx = np.empty(numleaves) # x-coordinate of pixel at centroid of each leaf
    centroidy = np.empty(numleaves) # y-coordinate of pixel at centroid of each leaf
    newlabel = np.zeros(numleaves+1)
    j = 0
    pixelheight, pixelwidth = np.shape(labeled)
    for i in range(1, numleaves+1):
        leafpixels = np.sum((labeled==i), dtype=np.int32)
        if(leafpixels >= minsegmentpixels):  # keep only if region is large enough
            j += 1
            newlabel[i] = j
            leafarea.append(float(leafpixels) * scale * scale)

    # Remove areas that are smaller than segmentationthreshold cm^2 (or inches^2)
    replace = {}
    for i in range(1, numleaves+1):
        replace.update({i: newlabel[i]})
#    print "replace = |%s|" % str(replace)
#    print sizes
#    pylab.imshow(labeled)
    newlabeled = np.copy(labeled)
    for key, value in replace.iteritems():
        newlabeled[labeled==key] = value
    numleaves = j

#    print('Found {} leaves.'.format(numleaves))

    # Fill the holes
    leafimagefilled = {}
    dimtuple = grayleafimage.shape
    dimlist = list(dimtuple)
    dimlist.insert(0, numleaves)
    fillregion = np.zeros(tuple(dimlist)).astype(np.uint8)
    fillarea = {}
    for i in range(1, numleaves+1):
        leafimagefilled[i-1] = ndimage.binary_fill_holes(newlabeled == i)
        fillregion[i-1] = np.subtract(leafimagefilled[i-1], newlabeled == i)  # region of holes in i-th leaf
        fillarea[i-1] = float(np.sum(fillregion[i-1])) * scale * scale

 #  The i-th leaf, with the interior holes filled:
 #  np.sum(newlabeled == i, fillregion[i-1])

    # Find the centroid of each leaf
    for i in range(1, numleaves+1):
        row,column = np.where(newlabeled == i)
        centroidy[i-1] = np.mean(row)
        centroidx[i-1] = np.mean(column)
                       
#    if debug:
#        pylab.subplot(2,3,4)
#        pylab.imshow(newlabeled)

    # Create an image as feedback to the user, coloring leaves red and filled leaf areas blue.
    redmask = newlabeled > 0
#    print 'fillregion dimensions = ', fillregion.shape

    bluemask = np.sum(fillregion, axis=0)
    zeroimage = np.zeros(grayleafimage.shape).astype(np.uint8)
    leaves = np.transpose([redmask*(255-grayleafimage.astype(np.uint8)), zeroimage, zeroimage], (1, 2, 0))  # original leaves
#    print 'leaves min = ', np.amin(leaves), ', max = ', np.amax(leaves)
#    print 'leaves dimensions = ', leaves.shape
    fill = np.transpose([zeroimage, zeroimage, bluemask*(255-grayleafimage.astype(np.uint8))], (1, 2, 0))
    background = np.transpose(np.array([(1-redmask)*(1-bluemask)*colorleafimage[:,:,0].astype(np.uint8), (1-redmask)*(1-bluemask)*colorleafimage[:,:,1].astype(np.uint8), (1-redmask)*(1-bluemask)*colorleafimage[:,:,2].astype(np.uint8)]),(1,2,0))
#    print 'background min = ', np.amin(background), ', max = ', np.amax(background)
#    print 'background dimensions = ', background.shape
#    feedbackleaf = np.transpose(np.array([redmask*leafimage + (1-redmask)*colorleafimage[:,:,0],  (1-redmask)*colorleafimage[:,:,1],  (1-redmask)*colorleafimage[:,:,2]]), (1, 2, 0))
    feedbackleaf = leaves + fill + background
#    pylab.axis('off')
#    leafhandle = pylab.gcf()
#    if debug:
#        pylab.subplot(2,3,5)
#        pylab.imshow(feedbackleaf)

    font = ImageFont.truetype("arial.ttf", 96*max(m,n)/2048)
    fontwidth,fontheight = font.getsize('5')
    base = Image.fromarray(feedbackleaf.astype(np.uint8)).convert('RGBA')
    txt = Image.new('RGBA', base.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)

    for i in range(1, numleaves+1):  # display the leaf number at the centroid of each leaf
        draw.text((centroidx[i-1]-0.5*fontwidth, centroidy[i-1]-0.5*fontheight), str(i), font=font, fill=(255,255,255,255))       
    im = alpha_composite(txt, base)

    # Save this to a file for the user to review later
    pathname, filename = os.path.split(file)
    basename, ext = os.path.splitext(filename)
    feedbackimagefilename = os.path.realpath(pathname) + os.sep +  basename + '-feedback' + ext
#    with open(feedbackimagefilename, 'w') as outfile:
#        fig.canvas.print_png(outfile)
#    pylab.savefig(feedbackimagefilename, bbox_inches='tight', pad_inches=0)
    im.save(feedbackimagefilename)
    del draw
#    misc.imsave(feedbackimagefilename, feedbackleaf)
#    pylab.savefig(feedbackimagefilename, bbox_inches='tight', pad_inches=0, dpi=(m/leafsize[0]))
#    pylab.close()
      
    return(newlabeled, leafarea, fillarea, numleaves)
    
    
def alpha_composite(front, back):
    """Alpha composite two RGBA images.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object

    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result