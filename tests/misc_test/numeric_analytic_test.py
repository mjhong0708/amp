#!/usr/bin/env python
"""This test randomly generates data with the EMT potential in MD simulations,
and then checks for consistency between analytical and numerical forces,
as well as dloss_dparameters."""

from ase.calculators.emt import EMT
from ase.build import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from amp.regression import Regressor


def generate_data(count):
    """Generates test or training data with a simple MD simulation."""
    atoms = fcc110('Pt', (2, 2, 1), vacuum=7.)
    adsorbate = Atoms([Atom('Cu', atoms[3].position + (0., 0., 2.5)),
                       Atom('Cu', atoms[3].position + (0., 0., 5.))])
    atoms.extend(adsorbate)
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    atoms.set_calculator(EMT())
    MaxwellBoltzmannDistribution(atoms, 300. * units.kB)
    dyn = VelocityVerlet(atoms, dt=1. * units.fs)
    newatoms = atoms.copy()
    newatoms.set_calculator(EMT())
    newatoms.get_potential_energy()
    images = [newatoms]
    for step in range(count - 1):
        dyn.run(50)
        newatoms = atoms.copy()
        newatoms.set_calculator(EMT())
        newatoms.get_potential_energy()
        del newatoms.constraints  # See ASE issue #64.
        images.append(newatoms)
    return images


def test():
    """Gaussian/Neural numeric-analytic consistency."""
    images = generate_data(2)
    regressor = Regressor(optimizer='BFGS')

    calc = Amp(descriptor=Gaussian(),
               model=NeuralNetwork(hiddenlayers=(3, 3),
                                   regressor=regressor,),
               cores=1)

    step = 0
    for d in [None, 0.00001]:
        for fortran in [True, False]:
            for cores in [1, 2]:
                step += 1
                label = \
                    'numeric_analytic_test/analytic-%s-%i' % (fortran, cores) \
                    if d is None \
                    else 'numeric_analytic_test/numeric-%s-%i' \
                    % (fortran, cores)
                print(label)

                loss = LossFunction(convergence={'energy_rmse': 10 ** 10,
                                                 'force_rmse': 10 ** 10},
                                    d=d)
                calc.set_label(label)
                calc.dblabel = 'numeric_analytic_test/analytic-True-1'
                calc.model.lossfunction = loss
                calc.descriptor.fortran = fortran
                calc.model.fortran = fortran
                calc.cores = cores

                calc.train(images=images,)

                if step == 1:
                    ref_energies = []
                    ref_forces = []
                    for image in images:
                        ref_energies += [calc.get_potential_energy(image)]
                        ref_forces += [calc.get_forces(image)]
                        ref_dloss_dparameters = \
                            calc.model.lossfunction.dloss_dparameters
                else:
                    energies = []
                    forces = []
                    for image in images:
                        energies += [calc.get_potential_energy(image)]
                        forces += [calc.get_forces(image)]
                        dloss_dparameters = \
                            calc.model.lossfunction.dloss_dparameters

                    for image_no in range(2):

                        diff = abs(energies[image_no] - ref_energies[image_no])
                        assert (diff < 10.**(-13.)), \
                            'The calculated value of energy of image %i is ' \
                            'wrong!' % (image_no + 1)

                        for atom_no in range(6):
                            for i in range(3):
                                diff = abs(forces[image_no][atom_no][i] -
                                           ref_forces[image_no][atom_no][i])
                                assert (diff < 10.**(-10.)), \
                                    'The calculated %i force of atom %i of ' \
                                    'image %i is wrong!' \
                                    % (i, atom_no, image_no + 1)
                        # Checks analytical and numerical dloss_dparameters
                        for _ in range(len(ref_dloss_dparameters)):
                            diff = abs(dloss_dparameters[_] -
                                       ref_dloss_dparameters[_])
                            assert(diff < 10 ** (-10.)), \
                                'The calculated value of loss function ' \
                                'derivative is wrong!'
    # Checks analytical and numerical forces
    forces = []
    for image in images:
        image.set_calculator(calc)
        forces += [calc.calculate_numerical_forces(image, d=d)]
    for atom_no in range(6):
        for i in range(3):
            diff = abs(forces[image_no][atom_no][i] -
                       ref_forces[image_no][atom_no][i])
            print("diff =", diff)
            assert (diff < 10.**(-6.)), \
                'The calculated %i force of atom %i of ' \
                'image %i is wrong! (Diff = %f)' \
                % (i, atom_no, image_no + 1, diff)

if __name__ == '__main__':
    test()
