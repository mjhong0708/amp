#!/usr/bin/env python
"""Test that the neural network regression works in training on energy only
both with none and behler fingerprint.
At some point in the development, this ability was temporarily lost!
"""

import os
from ase.calculators.emt import EMT
from ase.lattice.surface import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
from amp import Amp
from amp.fingerprint import Behler

###############################################################################


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
    newatoms.get_potential_energy(apply_constraint=False)
    images = [newatoms]
    for step in range(count):
        dyn.run(5)
        newatoms = atoms.copy()
        newatoms.set_calculator(EMT())
        newatoms.get_potential_energy(apply_constraint=False)
        images.append(newatoms)
    return images

###############################################################################


def test_none():
    label = 'noforce_test'
    if not os.path.exists(label):
        os.mkdir(label)

    print('Generating data.')
    images = generate_data(10)

    print('Training none-neural network.')
    calc = Amp(fingerprint=None, label=os.path.join(label, 'none'))
    calc.train(images, force_goal=None)

###############################################################################


def test_behler():
    label = 'noforce_test'

    print('Generating data.')
    images = generate_data(10)

    print('Training behler-neural network.')
    calc = Amp(fingerprint=Behler(), label=os.path.join(label, 'behler'))
    calc.train(images, force_goal=None)

###############################################################################

if __name__ == '__main__':
    test_none()
    test_behler()
