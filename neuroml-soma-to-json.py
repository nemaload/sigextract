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



if __name__ == '__main__':
    nmdir = sys.argv[1]
    neurons = nmllib.load_neurons(nmdir)
    print nmllib.jsondump_neurons(neurons)
