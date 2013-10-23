# NeuroML2 tools library

import glob
import numpy

import neuroml
import neuroml.loaders as loaders

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

def load_neurons(nmdir):
    neurons = []
    for nmfilename in glob.glob(nmdir + '/*.nml'):
        for neuron in load_neuron(nmfilename):
            neurons.append(neuron)
    return neurons