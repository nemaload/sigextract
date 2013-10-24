# Library with tools for dealing with the imaged worm pose
#
# bb = backbone (a spline tracing the A-P axis of the worm)
#
# proj = transformation of neuron coordinates from idealized worm model
# (as stored in NeuroML) to position corresonding to the imaged worm pose
# (described by poseinfo map with keys "zoom", "shift", "angle")

import json
import math
import numpy
import os
import scipy.interpolate as interp


def bbReadTSV(f):
    """
    Read TSV data (as output e.g. by pose-extract-lf.py) from @f.
    Returns a tuple of (points, edgedists), each point is a 3D
    coordinate (in the z,y,x order).
    """
    points = []
    edgedists = []
    for line in f:
        items = line.strip().split()
        points.append(items[0:2])
        edgedists.append(item[3])
    return (points, edgedists)

def bbReadJSON(f):
    """
    Read backbone JSON data (as output e.g. by tsv2json) from @f.
    Returns a tuple of (points, edgedists), each point is a 3D
    coordinate (in the z,y,x order).
    """
    data = json.load(f)
    points = []
    edgedists = []
    for point in data["bbpoints"]:
        points.append([point[2], point[1], point[0]])
        edgedists.append(point[3])
    return (points, edgedists)

def bbLoad(bbfilename):
    """
    Load backbone information from a file in one of two supported formats.
    """
    bbext = os.path.splitext(bbfilename)[1]

    if bbext == '.tsv':
        bbfile = open(bbfilename, 'r')
        (points, edgedists) = bbReadTSV(bbfile)
    elif bbext == '.json':
        bbfile = open(bbfilename, 'r')
        (points, edgedists) = bbReadJSON(bbfile)
    else:
        raise ValueError('Unknown backbone data extension ' + bbext)

    # Possibly reverse the backbone - just for user friendliness wrt. debug
    # images
    if points[0][2] > points[-1][2]:
        points = points[::-1]
        edgedists = edgedists[::-1]

    return (points, edgedists)


def bbToSpline(points):
    """
    Convert a sequence of points to a scipy spline object.
    Return a ([spline_y, spline_x], bblength) tuple, where @bblength
    is an estimated number of pixels along the backbone spline.
    """

    def p2pDist(point0, point1):
        return math.sqrt(sum([(point0[i] - point1[i])**2 for i in range(len(point0))]))

    # point2point distances for spline x-axis
    p2pdists = [0.]
    for i in range(1, len(points)):
        p2pdists.append(p2pDist(points[i-1], points[i]))
    # normalize
    totdist = sum(p2pdists)
    rundist = 0
    for i in range(len(p2pdists)):
        rundist += p2pdists[i]
        p2pdists[i] = rundist / totdist

    xtck = interp.splrep(p2pdists, [p[1] for p in points])
    ytck = interp.splrep(p2pdists, [p[2] for p in points])
    return ([ytck, xtck], int(totdist))


def bbTraceSpline(spline, bbpixels, uvframe):
    """
    Convert a backbone spline to a list of coordinates of pixels belonging
    to the spline plus information about the backbone direction at that point.
    Returns a list of ([y, x], [dy, dx]) tuples.
    """
    ticker = numpy.arange(0, 1.01, 1/float(bbpixels))
    yy = interp.splev(ticker, spline[0], der=0)
    xx = interp.splev(ticker, spline[1], der=0)
    dyy = interp.splev(ticker, spline[0], der=1)
    dxx = interp.splev(ticker, spline[1], der=1)

    # Normalize the derivation to describe a step for one pixel, i.e.
    # so that dx'**2 + dy'**2 = 1
    # dx**2 + dy**2 = n --> dx**2/n + dy**2/n = 1 --> d?' = d?/sqrt(n)
    sn = numpy.sqrt(dyy * dyy + dxx * dxx)
    dyy /= sn
    dxx /= sn

    # We are rotated by 90 degrees, hence "wrong" order in target tuple.
    return numpy.array([
            ([x, y], [dx, dy])
                for (y, x, dy, dx) in zip(yy, xx, dyy, dxx)
        ])


def projCoord(pos, poseinfo):
    """
    Return xy 2D projection of @pos according to @poseinfo.
    This transforms the coordinates from idealized worm model (as stored
    in NeuroML) to position corresonding to the imaged worm pose, except
    bending by the bb which is a separate step.

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

def projDiameter(diam, poseinfo):
    """
    Return 2D projection of circle @diameter according to @poseinfo.
    """
    zoom = poseinfo["zoom"]
    return diam * zoom

def projTranslateByBb(coord, bbpoints, name, poseinfo):
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
