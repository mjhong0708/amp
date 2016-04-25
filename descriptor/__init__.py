#!/usr/bin/env python
"""
Folder that contains different local environment descriptors.

"""

from ase.calculators.neighborlist import NeighborList
from ase.calculators.calculator import Parameters


# Neighborlist Calculator
class NeighborlistCalculator:

    """For integration with .utilities.Data
    For each image fed to calculate, a list of neighbors with offset
    distances is returned.
    """

    def __init__(self, cutoff):
        self.globals = Parameters({'cutoff': cutoff})
        self.keyed = Parameters()
        self.parallel_command = 'calculate_neighborlists'

    def calculate(self, image, key):
        cutoff = self.globals.cutoff
        n = NeighborList(cutoffs=[cutoff / 2.] * len(image),
                         self_interaction=False,
                         bothways=True,
                         skin=0.)
        n.update(image)
        return [n.get_neighbors(index) for index in xrange(len(image))]
