#!/usr/bin/env python
#
# interpose-neuroml - a toy demo script that will display worm images
# with neuroml neuron positions interposed visually (relative to the
# backbone and extra transformations parametrized manually)
#
# Usage: interpose-neuroml.py HDF5FILE FRAMENUMBER BACKBONEFILE POSEINFO NEUROML2DIR
#
# BACKBONEFILE file describes the backbone of the current worm pose.
#
# POSEINFO is the current worm pose description in the format
#   ZOOM,SHIFT,ANGLE
# where ZOOM is the multiplicative element for both directions
# (possibly negative to reverse the direction of the worm),
# SHIFT is the coordinate of the point in the spine which lies
# in the horizontal middle of the worm (i.e. when we see just the
# back tip of the worm, it will be a huge negative number),
# and ANGLE is the rotation angle around the A-P axis of the worm
# (in degrees).
#
# NEUROML2DIR is a directory containing NeuroML2 XML files (.nml)
# describing the cells to be shown. The positions stored in the files
# have to be based on a straightened worm model! (Produced e.g. by
# openworm/CElegansNeuroML:CElegans/pythonScripts/PositionStraighten.py)

import math
import sys
import tables

import numpy
import hdf5lflib
import poselib
import nmllib

import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.path import Path


def draw_uvframe_neurons(uvframe, bbpoints, neurons, poseinfo, title):
    # Draw the image with neuron locations interposed
    f = plt.figure(title)
    imgplot = plt.imshow(uvframe, cmap=plt.cm.gray)
    ax = f.add_subplot(111)

    for p in bbpoints:
        pos = [p[0,1], p[0,0]]
        ax.add_patch(matplotlib.patches.Circle(pos, radius = 0.5,
            edgecolor = 'yellow', fill = 0))

    for n in neurons:
        pos = poselib.projTranslateByBb(poselib.projCoord(n["pos"], poseinfo), bbpoints, n["name"], poseinfo)
        if pos is None:
            continue
        r = poselib.projDiameter(n["diameter"], poseinfo) / 2.
        print "showing", n["name"], "pos", pos, "r", r
        ax.add_patch(matplotlib.patches.Circle(pos, radius = r / 10.,
            edgecolor = 'green', fill = 0))
        ax.annotate(n["name"], xy = pos, color = 'green')

    plt.show()


if __name__ == '__main__':
    filename = sys.argv[1]
    frameNo = int(sys.argv[2])
    bbfilename = sys.argv[3]
    poseinfo_str = sys.argv[4]
    poseinfo = dict(zip(["zoom", "shift", "angle"], [float(f) for f in poseinfo_str.split(',')]))
    nmdir = sys.argv[5]

    # Load the image uvframe
    h5file = tables.open_file(filename, mode = "r")
    node = h5file.get_node('/', '/images/' + str(frameNo))
    ar = h5file.get_node('/', '/autorectification')
    try:
        cw = h5file.get_node('/', '/cropwindow')
    except tables.NoSuchNodeError:
        cw = None
    uvframe = hdf5lflib.compute_uvframe(node, ar, cw)

    # Load the backbone spline
    (points, edgedists) = poselib.bbLoad(bbfilename)
    (spline, bblength) = poselib.bbToSpline(points)
    bbpoints = poselib.bbTraceSpline(spline, bblength, uvframe)

    # Load neuron positions
    neurons = nmllib.load_neurons(nmdir)

    draw_uvframe_neurons(uvframe, bbpoints, neurons, poseinfo, poseinfo_str)
