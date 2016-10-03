#!/usr/bin/env python

from . import Amp
from .utilities import now, hash_images, make_filename, Logger
import os
import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot
rcParams.update({'figure.autolayout': True})


def plot_parity(load,
                images,
                label='parity',
                dblabel=None,
                plot_forces=True,
                plotfile=None,
                color='b.',
                cores=None,
                overwrite=False,
                returndata=False):
    """
    Makes a parity plot of Amp energies and forces versus real energies and
    forces.

    :param load: Path for loading an existing ".amp" file. Should be fed like
                 'load="filename.amp"'.
    :type load: str
    :param images: List of ASE atoms objects with positions, symbols, energies,
                   and forces in ASE format. This can also be the path to an
                   ASE trajectory (.traj) or database (.db) file.
                   Energies can be obtained from any reference, e.g. DFT
                   calculations.
    :type images: list or str
    :param label: Default prefix/location used for all files.
    :type label: str
    :param dblabel: Optional separate prefix/location of database files,
                    including fingerprints, fingerprint primes, and
                    neighborlists, to avoid calculating them. If not supplied,
                    just uses the value from label.
    :type dblabel: str
    :param plot_forces: Determines whether or not forces should be plotted as
                        well.
    :type plot_forces: bool
    :param plotfile: File for plots.
    :type plotfile: Object
    :param color: Plot color.
    :type color: str
    :param cores: Can specify cores to use for parallel training;
                  if None, will determine from environment
    :type cores: int
    :param overwrite: If a plot or an script containing values found overwrite
                      it.
    :type overwrite: bool
    :param returndata: Whether to return a reference to the figures and their
                       data or not.
    :type returndata: bool
    """

    if plotfile is None:
        plotfile = make_filename(label, 'plot.pdf')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile)

    log = Logger(make_filename(label, '-log.txt'))

    calc = Amp.load(file=load)
    calc.cores = cores
    calc.dblabel = label if dblabel is None else dblabel

    log('\nAmp parity plot started. ' + now() + '\n')
    log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images, log=log)

    log('\nDescriptor\n==========')
    # Derivatives of fingerprints need to be calculated if plot_forces is True.
    calc.descriptor.calculate_fingerprints(
        images=images,
        cores=calc.cores,
        log=log,
        calculate_derivatives=plot_forces)

    log('Calculating potential energies...', tic='pot-energy')
    energy_data = {}
    for hash, image in images.iteritems():
        amp_energy = calc.model.get_energy(calc.descriptor.fingerprints[hash])
        actual_energy = image.get_potential_energy(apply_constraint=False)
        energy_data[hash] = [actual_energy, amp_energy]
    log('...potential energies calculated.', toc='pot-energy')

    min_act_energy = min([energy_data[hash][0]
                         for hash, image in images.iteritems()])
    max_act_energy = max([energy_data[hash][0]
                         for hash, image in images.iteritems()])

    if plot_forces is False:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    log('Plotting energies...', tic='energy-plot')
    for hash, image in images.iteritems():
        ax.plot(energy_data[hash][0], energy_data[hash][1], color)
    # draw line
    ax.plot([min_act_energy, max_act_energy],
            [min_act_energy, max_act_energy],
            'r-',
            lw=0.3,)
    ax.set_xlabel("ab initio energy, eV")
    ax.set_ylabel("Amp energy, eV")
    ax.set_title("Energies")
    log('...energies plotted.', toc='energy-plot')

    if plot_forces is True:
        ax = fig.add_subplot(212)

        log('Calculating forces...', tic='forces')
        force_data = {}
        for hash, image in images.iteritems():
            amp_forces = \
                calc.model.get_forces(
                    calc.descriptor.fingerprints[hash],
                    calc.descriptor.fingerprintprimes[hash])
            actual_forces = image.get_forces(apply_constraint=False)
            force_data[hash] = [actual_forces, amp_forces]
        log('...forces calculated.', toc='forces')

        min_act_force = min([force_data[hash][0][index][k]
                            for hash, image in images.iteritems()
                            for index in range(len(image))
                            for k in range(3)])

        max_act_force = max([force_data[hash][0][index][k]
                            for hash, image in images.iteritems()
                            for index in range(len(image))
                            for k in range(3)])

        log('Plotting forces...', tic='force-plot')
        for hash, image in images.iteritems():
            for index in range(len(image)):
                for k in range(3):
                    ax.plot(force_data[hash][0][index][k],
                            force_data[hash][1][index][k], color)
        # draw line
        ax.plot([min_act_force, max_act_force],
                [min_act_force, max_act_force],
                'r-',
                lw=0.3,)
        ax.set_xlabel("ab initio force, eV/Ang")
        ax.set_ylabel("Amp force, eV/Ang")
        ax.set_title("Forces")
        log('...forces plotted.', toc='force-plot')

    fig.savefig(plotfile)

    if returndata:
        if plot_forces is False:
            return fig, energy_data
        else:
            return fig, energy_data, force_data


def plot_error(load,
               images,
               label='parity',
               dblabel=None,
               plot_forces=True,
               plotfile=None,
               color='b.',
               cores=None,
               overwrite=False,
               returndata=False):
    """
    Makes a parity plot of Amp energies and forces versus real energies and
    forces.

    :param load: Path for loading an existing ".amp" file. Should be fed like
                 'load="filename.amp"'.
    :type load: str
    :param images: List of ASE atoms objects with positions, symbols, energies,
                   and forces in ASE format. This can also be the path to an
                   ASE trajectory (.traj) or database (.db) file.
                   Energies can be obtained from any reference, e.g. DFT
                   calculations.
    :type images: list or str
    :param label: Default prefix/location used for all files.
    :type label: str
    :param dblabel: Optional separate prefix/location of database files,
                    including fingerprints, fingerprint primes, and
                    neighborlists, to avoid calculating them. If not supplied,
                    just uses the value from label.
    :type dblabel: str
    :param plot_forces: Determines whether or not forces should be plotted as
                        well.
    :type plot_forces: bool
    :param plotfile: File for plots.
    :type plotfile: Object
    :param color: Plot color.
    :type color: str
    :param cores: Can specify cores to use for parallel training;
                  if None, will determine from environment
    :type cores: int
    :param overwrite: If a plot or an script containing values found overwrite
                      it.
    :type overwrite: bool
    :param returndata: Whether to return a reference to the figures and their
                       data or not.
    :type returndata: bool
    """

    if plotfile is None:
        plotfile = make_filename(label, 'plot.pdf')

    if (not overwrite) and os.path.exists(plotfile):
        raise IOError('File exists: %s.\nIf you want to overwrite,'
                      ' set overwrite=True or manually delete.'
                      % plotfile)

    log = Logger(make_filename(label, '-log.txt'))

    calc = Amp.load(file=load)
    calc.cores = cores
    calc.dblabel = label if dblabel is None else dblabel

    log('\nAmp error plot started. ' + now() + '\n')
    log('Descriptor: %s' % calc.descriptor.__class__.__name__)
    log('Model: %s' % calc.model.__class__.__name__)

    images = hash_images(images, log=log)

    log('\nDescriptor\n==========')
    # Derivatives of fingerprints need to be calculated if plot_forces is True.
    calc.descriptor.calculate_fingerprints(
        images=images,
        cores=calc.cores,
        log=log,
        calculate_derivatives=plot_forces)

    log('Calculating potential energy errors...', tic='pot-energy')
    energy_data = {}
    for hash, image in images.iteritems():
        no_of_atoms = len(image)
        amp_energy = calc.model.get_energy(calc.descriptor.fingerprints[hash])
        actual_energy = image.get_potential_energy(apply_constraint=False)
        act_energy_per_atom = actual_energy / no_of_atoms
        energy_error = abs(amp_energy - actual_energy) / no_of_atoms
        energy_data[hash] = [act_energy_per_atom, energy_error]
    log('...potential energy errors calculated.', toc='pot-energy')

    # calculating energy per atom rmse
    energy_square_error = 0.
    for hash, image in images.iteritems():
        energy_square_error += energy_data[hash][1] ** 2.
    energy_per_atom_rmse = np.sqrt(energy_square_error / len(images))

    min_act_energy_per_atom = min([energy_data[hash][0]
                                   for hash, image in images.iteritems()])
    max_act_energy_per_atom = max([energy_data[hash][0]
                                   for hash, image in images.iteritems()])

    if plot_forces is False:
        fig = pyplot.figure(figsize=(5., 5.))
        ax = fig.add_subplot(111)
    else:
        fig = pyplot.figure(figsize=(5., 10.))
        ax = fig.add_subplot(211)

    log('Plotting energy errors...', tic='energy-plot')
    for hash, image in images.iteritems():
        ax.plot(energy_data[hash][0], energy_data[hash][1], color)
    # draw horizontal line for rmse
    ax.plot([min_act_energy_per_atom, max_act_energy_per_atom],
            [energy_per_atom_rmse, energy_per_atom_rmse],
            color='black', linestyle='dashed', lw=1,)
    ax.text(max_act_energy_per_atom,
            energy_per_atom_rmse,
            'energy rmse = %6.5f' % energy_per_atom_rmse,
            ha='right',
            va='bottom',
            color='black')
    ax.set_xlabel("ab initio energy (eV) per atom")
    ax.set_ylabel("$|$ab initio energy - Amp energy$|$ / number of atoms")
    ax.set_title("Energies")
    log('...energy errors plotted.', toc='energy-plot')

    if plot_forces is True:
        ax = fig.add_subplot(212)

        log('Calculating force errors...', tic='forces')
        force_data = {}
        for hash, image in images.iteritems():
            amp_forces = \
                calc.model.get_forces(
                    calc.descriptor.fingerprints[hash],
                    calc.descriptor.fingerprintprimes[hash])
            actual_forces = image.get_forces(apply_constraint=False)
            force_data[hash] = [
                actual_forces,
                abs(np.array(amp_forces) - np.array(actual_forces))]
        log('...force errors calculated.', toc='forces')

        # calculating force rmse
        force_square_error = 0.
        for hash, image in images.iteritems():
            no_of_atoms = len(image)
            for index in range(no_of_atoms):
                for k in range(3):
                    force_square_error += \
                        ((1.0 / 3.0) * force_data[hash][1][index][k] ** 2.) / \
                        no_of_atoms
        force_rmse = np.sqrt(force_square_error / len(images))

        min_act_force = min([force_data[hash][0][index][k]
                            for hash, image in images.iteritems()
                            for index in range(len(image))
                            for k in range(3)])

        max_act_force = max([force_data[hash][0][index][k]
                            for hash, image in images.iteritems()
                            for index in range(len(image))
                            for k in range(3)])

        log('Plotting force errors...', tic='force-plot')
        for hash, image in images.iteritems():
            for index in range(len(image)):
                for k in range(3):
                    ax.plot(force_data[hash][0][index][k],
                            force_data[hash][1][index][k], color)
        # draw horizontal line for rmse
        ax.plot([min_act_force, max_act_force],
                [force_rmse, force_rmse],
                color='black',
                linestyle='dashed',
                lw=1,)
        ax.text(max_act_force,
                force_rmse,
                'force rmse = %5.4f' % force_rmse,
                ha='right',
                va='bottom',
                color='black',)
        ax.set_xlabel("ab initio force, eV/Ang")
        ax.set_ylabel("$|$ab initio force - Amp force$|$")
        ax.set_title("Forces")
        log('...force errors plotted.', toc='force-plot')

    fig.savefig(plotfile)

    if returndata:
        if plot_forces is False:
            return fig, energy_data
        else:
            return fig, energy_data, force_data
