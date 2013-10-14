#!/usr/bin/env python
#
# straighten - a toy demo script applying backbone information to optical
# straightening of the source image by slicing and restacking the image
#
# Usage: straighten.py HDF5FILE FRAMENUMBER BACKBONEFILE [OUTPUTFILE]
#
# If OUTPUTFILE is not passed, the straightening result is shown on screen.

import math
import random

import numpy
import hdf5lflib
import bblib

import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.misc

import os
import sys
import tables

def restackBySpline(spoints, uvframe, cpoints, edgedists):
    """
    Restack input pixel frame @uvframe by sequence of traced spline
    points @spoints while using the (@cpoints, @edgedists) data to
    determine the stopping point.
    """
    # Width of restacked frame is # of traced points
    # Height of restacked frame is the maximum edge distance * 2.
    (height, width) = (int(math.ceil(max(edgedists) * 2)), len(spoints))
    restackframe = numpy.zeros(shape=(height, width), dtype='short')
    basey = height / 2

    for x in range(width):
        # coordinates and derivation
        (c, d) = spoints[x]
        for y in range(basey-1):
            # venture perpendicularly from derivation
            coord = [c[0] - y*d[1], c[1] + y*d[0]]
            if coord[0] >= 0 and coord[1] >= 0:
                try:
                    restackframe[basey - y][x] = int(hdf5lflib.pointInterpolate(uvframe, coord))
                except IndexError:
                    pass
        for y in range(1, basey-1):
            coord = [c[0] + y*d[1], c[1] - y*d[0]]
            if coord[0] >= 0 and coord[1] >= 0:
                try:
                    restackframe[basey + y][x] = int(hdf5lflib.pointInterpolate(uvframe, coord))
                except IndexError:
                    pass

    return restackframe

if __name__ == '__main__':
    filename = sys.argv[1]
    frameNo = int(sys.argv[2])
    bbfilename = sys.argv[3]
    outputfile = None
    if len(sys.argv) >= 5:
        outputfile = sys.argv[4]

    h5file = tables.open_file(filename, mode = "r")
    node = h5file.get_node('/', '/images/' + str(frameNo))
    ar = h5file.get_node('/', '/autorectification')
    try:
        cw = h5file.get_node('/', '/cropwindow')
    except tables.NoSuchNodeError:
        cw = None
    uvframe = hdf5lflib.compute_uvframe(node, ar, cw)

    (points, edgedists) = bblib.loadBackbone(bbfilename)

    (spline, bblength) = bblib.backboneSpline(points)
    bbpoints = bblib.traceBackbone(spline, bblength, uvframe)

    # Draw the backbone
    #plt.figure()
    #plt.plot(bbpoints[:,0,1], bbpoints[:,0,0], 'o') # (x, y) order
    #plt.axis([0,100,100,0])
    #plt.show()

    restackframe = restackBySpline(bbpoints, uvframe, points, edgedists)

    if outputfile:
        scipy.misc.imsave(outputfile, restackframe)
    else:
        # Draw the restacked image
        fig, axes = plt.subplots(ncols = 2)
        axes[0].imshow(uvframe, cmap=plt.cm.gray)
        axes[1].imshow(restackframe, cmap=plt.cm.gray)
        plt.show()
