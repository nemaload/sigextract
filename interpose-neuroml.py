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

import glob
import math
import sys
import tables

import numpy
import hdf5lflib
import bblib

import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.path import Path

import neuroml
import neuroml.loaders as loaders

def project_coord(pos, poseinfo):
    """
    Return xy 2D projection of @pos according to @poseinfo.

    @pos is in coordinate system:
    z ^ . x
      |/
      +--> y
    """

    # Apply zoom
    zoom = poseinfo["zoom"]
    pos = pos * zoom

    # Apply rotation (around the y axis)
    alpha = poseinfo["angle"] * math.pi / 180.
    d = math.sqrt(pos[0]**2 + pos[2]**2) # dist from 0
    beta = math.asin(pos[2] / d) # current angle
    gamma = alpha + beta # new angle
    pos[0] = d * math.cos(gamma)
    pos[2] = d * math.sin(gamma)

    # Flatten - ignore the x coordinate ("depth")
    return (pos[1], pos[2])


def project_diameter(diam, poseinfo):
    """
    Return 2D projection of circle @diameter according to @poseinfo.
    """
    zoom = poseinfo["zoom"]
    return diam * zoom

def translate_by_bb(coord, bbpoints, name, poseinfo):
    """
    Translate xy @coord by the corresponding spine point of @bbpoints.
    The x coordinate determines a point _on_ the spine, the y coordinate
    then points perpendicularly.
    """
    coord_x = coord[0]
    coord_x += poseinfo["shift"]

    if coord_x < 0.:
        return None
    try:
        bbpoints0 = bbpoints[int(coord_x)]
        bbpoints1 = bbpoints[int(coord_x + 1.)]
    except IndexError:
        return None

    beta = coord_x - int(coord_x)
    (base_c, base_d) = bbpoints1 * beta + bbpoints0 * (1. - beta)

    c = base_c + coord[1] * base_d
    return [c[1], c[0]]


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
        pos = translate_by_bb(project_coord(n["pos"], poseinfo), bbpoints, n["name"], poseinfo)
        if pos is None:
            continue
        r = project_diameter(n["diameter"], poseinfo) / 2.
        print "showing", n["name"], "pos", pos, "r", r
        ax.add_patch(matplotlib.patches.Circle(pos, radius = r / 10.,
            edgecolor = 'green', fill = 0))
        ax.annotate(n["name"], xy = pos, color = 'green')

    plt.show()


def load_neuron(filename):
    print "Loading " + filename + "..."
    doc = loaders.NeuroMLLoader.load(filename)
    neurons = []
    for cell in doc.cells:
        soma_segid = filter(lambda g: g.id == "Soma", cell.morphology.segment_groups)[0].members[0].segments
        segment = cell.morphology.segments[soma_segid]
        print "  Loading cell " + cell.id
        # Normally, proximal and distal coordinates will be the same in our
        # dataset; if not, average them just to be sure; then consider a circle
        # around this coordinate to be the neuron location.
        neurons.append({
                'name': cell.id,
                'pos': numpy.array([
                    (segment.proximal.x + segment.distal.x) / 2.,
                    (segment.proximal.y + segment.distal.y) / 2.,
                    (segment.proximal.z + segment.distal.z) / 2.]),
                'diameter': (segment.proximal.diameter + segment.distal.diameter) / 2.,
            })
    return neurons

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
    (points, edgedists) = bblib.loadBackbone(bbfilename)
    (spline, bblength) = bblib.backboneSpline(points)
    bbpoints = bblib.traceBackbone(spline, bblength, uvframe)

    # Load neuron positions
    neurons = []
    for nmfilename in glob.glob(nmdir + '/*.nml'):
        for neuron in load_neuron(nmfilename):
            neurons.append(neuron)

    draw_uvframe_neurons(uvframe, bbpoints, neurons, poseinfo, poseinfo_str)
