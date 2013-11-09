#!/usr/bin/env python
#
# Usage: bb-reverse.py FILENAME
#
# Load backbone and print it to the stdout (as TSV) reversed.
#
# Example: for i in *backbone*.json; do ./bb-reverse.py $i >$i.tsv; ./tsv2json.sh $i.tsv >$i; done

import poselib
import numpy
import sys


def printTSV(backbone, edgedists):
    for i in range(len(backbone)):
        point = backbone[i]
        print point[0], point[1], point[2], edgedists[i]

if __name__ == '__main__':
    filename = sys.argv[1]

    (points, edgedists) = poselib.bbLoad(filename)
    printTSV(points[::-1], edgedists[::-1])
