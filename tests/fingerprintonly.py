#!/usr/bin/env python
"""Test of the BP neural network calculator. Randomly generates data
with the EMT potential in MD simulations. Both trains and tests getting
energy out of the calculator. Shows results for both interpolation and
extrapolation."""

import os

from ase.calculators.emt import EMT
from ase.lattice.surface import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

from ampmoremodular import Amp
from ampmoremodular.descriptor import Behler
from ampmoremodular.regression import NeuralNetwork
from ampmoremodular.utilities import randomize_images, Data


def generate_data(count):
    """Generates test or training data with a simple MD simulation."""
    atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)
    adsorbate = Atoms([Atom('Cu', atoms[7].position + (0., 0., 2.5)),
                       Atom('Cu', atoms[7].position + (0., 0., 5.))])
    atoms.extend(adsorbate)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    atoms.set_calculator(EMT())
    MaxwellBoltzmannDistribution(atoms, 300. * units.kB)
    dyn = VelocityVerlet(atoms, dt=1. * units.fs)
    newatoms = atoms.copy()
    newatoms.set_calculator(EMT())
    newatoms.get_potential_energy()
    images = [newatoms]
    for step in range(count):
        dyn.run(5)
        newatoms = atoms.copy()
        newatoms.set_calculator(EMT())
        newatoms.get_potential_energy()
        images.append(newatoms)
    return images


label = 'amp/calc'
if not os.path.exists('amp'):
    os.mkdir('amp')

print('Generating data.')
all_images = generate_data(10)
train_images, test_images = randomize_images(all_images)

print('Training network.')
calc = Amp(label=label,
           descriptor=Behler(),
           regression=NeuralNetwork(hiddenlayers=(5, 5)),
           fortran=False,
           cores=1)

calc.fingerprint(images=train_images)


print('Interact with fingerprints')
fp = Data(filename=label + '-fingerprints')


