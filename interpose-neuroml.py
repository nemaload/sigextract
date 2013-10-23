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
#   ZOOM (TODO more)
# where ZOOM may be negative to reverse the direction of the worm.
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
    """
    zoom = poseinfo["zoom"]
    return (pos[1] * zoom, pos[2] * zoom)

def project_diameter(diam, poseinfo):
    """
    Return 2D projection of circle @diameter according to @poseinfo.
    """
    zoom = poseinfo["zoom"]
    return diam * zoom

def translate_by_bb(coord, bbpoints):
    """
    Translate xy @coord by the corresponding spine point of @bbpoints.
    The x coordinate determines a point _on_ the spine, the y coordinate
    then points perpendicularly.
    """
    try:
        bbpoints0 = bbpoints[int(coord[0])]
        bbpoints1 = bbpoints[int(coord[0] + 1.)]
    except IndexError:
        return None
    beta = coord[0] - int(coord[0])
    (base_c, base_d) = bbpoints1 * beta + bbpoints0 * (1. - beta)

    coord = base_c + coord[1] * base_d
    return [coord[1], coord[0]]


def draw_uvframe_neurons(uvframe, bbpoints, neurons, poseinfo):
    # Draw the image with neuron locations interposed
    f = plt.figure()
    imgplot = plt.imshow(uvframe, cmap=plt.cm.gray)
    ax = f.add_subplot(111)

    for p in bbpoints:
        pos = [p[0,1], p[0,0]]
        ax.add_patch(matplotlib.patches.Circle(pos, radius = 0.5,
            edgecolor = 'yellow', fill = 0))

    for n in neurons:
        pos = translate_by_bb(project_coord(n["pos"], poseinfo), bbpoints)
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
    poseinfo = { 'zoom': float(sys.argv[4]) } #dict(zip("zoom", sys.argv[3].split(','))
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

    draw_uvframe_neurons(uvframe, bbpoints, neurons, poseinfo)
