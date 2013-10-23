#!/usr/bin/env python
#
# neuroml-soma-to-json - convert set of neuroml neuron records to json data
#
# Usage: interpose-neuroml.py NEUROML2DIR
#
# NEUROML2DIR is a directory containing NeuroML2 XML files (.nml)
# describing the cells to be shown. The positions stored in the files
# have to be based on a straightened worm model! (Produced e.g. by
# openworm/CElegansNeuroML:CElegans/pythonScripts/PositionStraighten.py)

import json
import nmllib
import numpy
import sys


class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    nmdir = sys.argv[1]
    neurons = nmllib.load_neurons(nmdir)
    print json.dumps({"neurons": neurons}, cls = NumPyArangeEncoder)
