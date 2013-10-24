# Library with tools for dealing with the imaged worm pose
# bb = backbone (a spline tracing the A-P axis of the worm)

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
