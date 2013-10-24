# NeuroML2 tools library

from __future__ import print_function

import glob
import json
import numpy
import sys

import neuroml
import neuroml.loaders as loaders

def load_neurons_json(nmfile):
    f = open(nmfile, 'r')
    data = json.load(f)
    for n in data["neurons"]:
        n["pos"] = numpy.array(n["pos"])
    return data["neurons"]


def load_neuron(filename):
    print("Loading " + filename + "...", file = sys.stderr)
    doc = loaders.NeuroMLLoader.load(filename)
    neurons = []
    for cell in doc.cells:
        soma_segid = filter(lambda g: g.id == "Soma", cell.morphology.segment_groups)[0].members[0].segments
        segment = cell.morphology.segments[soma_segid]
        print("  Loading cell " + cell.id, file = sys.stderr)
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

def load_neurons_from_dir(nmdir):
    neurons = []
    for nmfilename in glob.glob(nmdir + '/*.nml'):
        for neuron in load_neuron(nmfilename):
            neurons.append(neuron)
    return neurons


def load_neurons(nmloc):
    if nmloc.endswith('.json'):
        return load_neurons_json(nmloc)
    else:
        return load_neurons_from_dir(nmloc)
