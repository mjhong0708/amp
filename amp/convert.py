import os
import shutil
import tempfile
import warnings


def save_to_prophet(calc, filename='potential_', overwrite=False,
                    units="metal"):
    """Saves the calculator in a way that it can be used with PROPhet.

    Parameters
    ----------
    calc : obj
        A trained Amp calculator object.
    filename : str
        File object or path to the file to write to.
    overwrite : bool
        If an output file with the same name exists, overwrite it.
    units : str
        LAMMPS units style to be used with the outfile file.
    """

    from ase.calculators.lammpslib import unit_convert

    warnings.warn(
        'Conversion from Amp to PROPhet leads to energies and forces being '
        'calculated correctly to within machine precision. Some choices of '
        'symmetry function parameters have been found to result in the two '
        'codes giving unequal energies and forces. It is important to verify '
        'that the two codes give equal energies and forces for your system '
        'prior to using PROPhet for MD.'
        '\n******************************************************************'
        '\nTo mitigate the problem:'
        '\n- It is recommended to use large `eta` for G2.'
        '\n  and large `eta` and `zeta` for G4.'
        '\n- Use PROPhet to retrain the NN for few iterations.'
        '\n- Test your results carefully'
        '\n******************************************************************')

    if os.path.exists(filename):
        if overwrite is False:
            oldfilename = filename
            filename = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                   suffix='.amp').name
            calc._log('File "%s" exists. Instead saving to "%s".' %
                      (oldfilename, filename))
        else:
            oldfilename = tempfile.NamedTemporaryFile(mode='w',
                                                      delete=False,
                                                      suffix='.amp').name

            calc._log('Overwriting file: "%s". Moving original to "%s".'
                      % (filename, oldfilename))
            shutil.move(filename, oldfilename)

    desc_pars = calc.descriptor.parameters
    model_pars = calc.model.parameters
    if (desc_pars['mode'] != 'atom-centered' or
       model_pars['mode'] != 'atom-centered'):
        raise NotImplementedError(
            'PROPhet requires atom-centered symmetry functions.')
    if desc_pars['cutoff']['name'] != 'Cosine':
        raise NotImplementedError(
            'PROPhet requires cosine cutoff functions.')
    if model_pars['activation'] != 'tanh':
        raise NotImplementedError(
            'PROPhet requires tanh activation functions.')
    els = desc_pars['elements']
    n_els = len(els)
    length_G2 = int(n_els)
    length_G4 = int(n_els*(n_els+1)/2)
    cutoff = (desc_pars['cutoff']['kwargs']['Rc'] /
              unit_convert('distance', units))
    # Get correct order of elements listed in the Amp object
    el = desc_pars['elements'][0]
    n_G2 = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G2')
    n_G4 = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G4')
    els_ordered = []
    if n_G2 > 0:
        for Gs in range(n_els):
            els_ordered.append(desc_pars['Gs'][el][Gs]['element'])
    elif n_G4 > 0:
        for Gs in range(n_els):
            els_ordered.append(desc_pars['Gs'][el][Gs]['elements'][0])
    else:
        raise RuntimeError('There must be at least one G2 or G4 symmetry '
                           'function.')
    # Write each element's PROPhet input file
    for el in desc_pars['elements']:
        f = open(filename + el, 'w')
        # Write header.
        f.write('nn\n')
        f.write('structure\n')
        # Write elements.
        f.write(el + ':  ')
        for el_i in els_ordered:
            f.write(el_i+' ')
        f.write('\n')
        n_G2_el = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G2')
        n_G4_el = sum(1 for k in desc_pars['Gs'][el] if k['type'] == 'G4')
        if n_G2_el != n_G2 or n_G4_el != n_G4:
            raise NotImplementedError(
                'PROPhet requires each element to have the same number of '
                'symmetry functions.')
        f.write(str(int(n_G2/length_G2+n_G4/length_G4))+'\n')
        # Write G2s.
        for Gs in range(0, n_G2, length_G2):
            eta = desc_pars['Gs'][el][Gs]['eta']
            for i in range(length_G2):
                eta_2 = desc_pars['Gs'][el][Gs+i]['eta']
                if eta != eta_2:
                    raise NotImplementedError(
                        'PROPhet requires each G2 function to have the '
                        'same eta value for all element pairs.')
            f.write('G2 ' + str(cutoff) + ' 0 ' + str(eta/cutoff**2) +
                    ' 0\n')
        # Write G4s (G3s in PROPhet).
        for Gs in range(n_G2, n_G2+n_G4, length_G4):
            eta = desc_pars['Gs'][el][Gs]['eta']
            gamma = desc_pars['Gs'][el][Gs]['gamma']
            zeta = desc_pars['Gs'][el][Gs]['zeta']
            for i in range(length_G4):
                eta_2 = desc_pars['Gs'][el][Gs+i]['eta']
                gamma_2 = desc_pars['Gs'][el][Gs+i]['gamma']
                zeta_2 = desc_pars['Gs'][el][Gs+i]['zeta']
                if eta != eta_2 or gamma != gamma_2 or zeta != zeta_2:
                    raise NotImplementedError(
                        'PROPhet requires each G4 function to have the '
                        'same eta, gamma, and zeta values for all '
                        'element pairs.')
            f.write('G3 ' + str(cutoff) + ' 0 ' + str(eta/cutoff**2) +
                    ' ' + str(zeta) + ' ' + str(gamma) + '\n')
        # Write input means for G2.
        for i in range(n_els):
            for Gs in range(0, n_G2, length_G2):
                # For debugging, to see the order of the PROPhet file
                # if el==desc_pars['elements'][0]:
                #    print(desc_pars['Gs'][el][Gs+i])
                mean = (model_pars['fprange'][el][Gs+i][1] +
                        model_pars['fprange'][el][Gs+i][0]) / 2.
                f.write(str(mean) + ' ')
        # Write input means for G4.
        for i in range(n_els):
            for j in range(n_els-i):
                for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                    # For debugging, to see the order of the PROPhet file
                    # if el==desc_pars['elements'][0]:
                    #    print(desc_pars['Gs'][el][Gs+j+n_els*i+int((i-i**2)/2)])
                    mean = (model_pars['fprange'][el][Gs + j + n_els * i +
                                                      int((i - i**2) / 2)][1] +
                            model_pars['fprange'][el][Gs + j + n_els * i +
                                                      int((i - i**2) / 2)][0])
                    # NB the G4 mean is doubled to correct for PROPhet
                    # counting each neighbor pair twice as much as Amp
                    f.write(str(mean) + ' ')
        f.write('\n')
        # Write input variances for G2.
        for i in range(n_els):
            for Gs in range(0, n_G2, length_G2):
                variance = (model_pars['fprange'][el][Gs+i][1] -
                            model_pars['fprange'][el][Gs+i][0]) / 2.
                f.write(str(variance) + ' ')
        # Write input variances for G4.
        for i in range(n_els):
            for j in range(n_els-i):
                for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                    variance = (model_pars['fprange'][el][Gs + j + n_els * i +
                                                          int((i - i**2) /
                                                              2)][1] -
                                model_pars['fprange'][el][Gs + j + n_els * i +
                                                          int((i - i**2) /
                                                              2)][0])
                    # NB the G4 variance is doubled to correct for PROPhet
                    # counting each neighbor pair twice as much as Amp
                    f.write(str(variance) + ' ')
        f.write('\n')
        f.write('energy\n')
        # Write output mean.
        f.write('0\n')
        # Write output variance.
        f.write('1\n')
        curr_node = 0
        # Write NN layer architecture.
        for nodes in model_pars['hiddenlayers'][el]:
            f.write(str(nodes)+' ')
        f.write('1\n')
        # Write first hidden layer of the NN for the symmetry functions.
        layer = 0
        f.write('[[ layer ' + str(layer) + ' ]]\n')
        for node in range(model_pars['hiddenlayers'][el][layer]):
            # Write each node of the layer.
            f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
            f.write('   ')
            # G2
            for i in range(n_els):
                for Gs in range(0, n_G2, length_G2):
                    f.write(str(model_pars['weights'][el]
                                [layer + 1][Gs + i][node]))
                    f.write('     ')
            # G4
            for i in range(n_els):
                for j in range(n_els-i):
                    for Gs in range(n_G2, n_G2 + n_G4, length_G4):
                        f.write(str(model_pars['weights'][el]
                                    [layer + 1][Gs + j + n_els * i +
                                                int((i - i**2) / 2)][node]))
                        f.write('     ')
            f.write('\n')
            f.write('   ')
            f.write(str(model_pars['weights'][el][layer+1][-1][node]))
            f.write('\n')
            curr_node += 1
        # Write remaining hidden layers of the NN.
        for layer in range(1, len(model_pars['hiddenlayers'][el])):
            f.write('[[ layer ' + str(layer) + ' ]]\n')
            for node in range(model_pars['hiddenlayers'][el][layer]):
                f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
                f.write('   ')
                for i in range(len(model_pars['weights'][el][layer+1])-1):
                    f.write(str(model_pars['weights'][el][layer+1][i][node]))
                    f.write('     ')
                f.write('\n')
                f.write('   ')
                f.write(str(model_pars['weights'][el][layer+1][-1][node]))
                f.write('\n')
                curr_node += 1
        # Write output layer of the NN, consisting of an activated node.
        f.write('[[ layer ' + str(layer+1) + ' ]]\n')
        f.write('  [ node ' + str(curr_node) + ' ]  tanh\n')
        f.write('   ')
        for i in range(len(model_pars['weights'][el][layer+2])-1):
            f.write(str(model_pars['weights'][el][layer+2][i][0]))
            f.write('     ')
        f.write('\n')
        f.write('   ')
        f.write(str(model_pars['weights'][el][layer+2][-1][0]))
        f.write('\n')
        curr_node += 1
        # Write output layer of the NN, consisting of a linear node,
        # representing Amp's scaling.
        f.write('[[ layer ' + str(layer+2) + ' ]]\n')
        f.write('  [ node ' + str(curr_node) + ' ]  linear\n')
        f.write('   ')
        f.write(str(model_pars['scalings'][el]['slope'] /
                    unit_convert('energy', units)))
        f.write('\n')
        f.write('   ')
        f.write(str(model_pars['scalings'][el]['intercept'] /
                    unit_convert('energy', units)))
        f.write('\n')
        f.close()


def save_to_openkim(calc, filename='amp.params', overwrite=False,
                    units="metal"):
    """Saves the calculator in a way that it can be used with OpenKIM.

    Parameters
    ----------
    calc : obj
        A trained Amp calculator object.
    filename : str
        File object or path to the file to write to.
    overwrite : bool
        If an output file with the same name exists, overwrite it.
    units : str
        LAMMPS units style to be used with the outfile file.
    """

    from ase.calculators.lammpslib import unit_convert

    if os.path.exists(filename):
        if overwrite is False:
            oldfilename = filename
            filename = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                   suffix='.params')
            calc._log('File "%s" exists. Instead saving to "%s".' %
                      (oldfilename, filename))
        else:
            oldfilename = tempfile.NamedTemporaryFile(mode='w',
                                                      delete=False,
                                                      suffix='.params')

            calc._log('Overwriting file: "%s". Moving original to "%s".'
                      % (filename, oldfilename))
            shutil.move(filename, oldfilename)

    desc_pars = calc.descriptor.parameters
    model_pars = calc.model.parameters
    if (desc_pars['mode'] != 'atom-centered' or
       model_pars['mode'] != 'atom-centered'):
        raise NotImplementedError(
            'KIM model requires atom-centered symmetry functions.')
    if desc_pars['cutoff']['name'] != 'Cosine':
        raise NotImplementedError(
            'KIM model requires cosine cutoff functions.')
    elements = desc_pars['elements']
#    path = os.path.dirname(__file__)
    elements = sorted(elements)
#    f = open(path + '/../tools/amp-kim/amp_parameterized_model/' +
#             filename, 'w')
    f = open(filename, 'w')
    f.write(str(len(elements)) + '  # number of chemical species')
    f.write('\n')
    f.write(' '.join(elements) + '  # chemical species')
    f.write('\n')
    f.write(' '.join(str(len(desc_pars['Gs'][element])) for element in
            elements) +
            '  # number of fingerprints of each chemical species')
    f.write('\n')
    for element in elements:
        count = 0
        # writing symmetry functions
        for G in desc_pars['Gs'][element]:
            if G['type'] == 'G2':
                f.write(element + ' ' + 'g2' + '  # fingerprint of %s' %
                        element)
                f.write('\n')
                f.write(G['element'] + ' ' + str(G['eta']) + '  # eta')
            elif G['type'] == 'G4':
                f.write(element + ' ' + 'g4' +
                        '  # fingerprint of %s' % element)
                f.write('\n')
                f.write(G['elements'][0] + ' ' + G['elements'][1] + ' ' +
                        str(G['eta']) + ' ' + str(G['gamma']) + ' ' +
                        str(G['zeta']) + '  # eta, gamma, zeta')
            f.write('\n')
            # writing fingerprint range
            f.write(str(model_pars['fprange'][element][count][0]) + ' ' +
                    str(model_pars['fprange'][element][count][1]) +
                    '  # range of fingerprint %i of %s' % (count, element))
            f.write('\n')
            count += 1
    # writing the cutoff
    cutoff = (desc_pars['cutoff']['kwargs']['Rc'] /
              unit_convert('distance', units))
    f.write(str(cutoff) + '  # cutoff radius')
    f.write('\n')
    f.write(model_pars['activation'] + '  # activation function')
    f.write('\n')
    # writing the neural network structures
    for element in elements:
        f.write(str(len(model_pars['hiddenlayers'][element])) +
                '  # number of hidden-layers of %s neural network' % element)
        f.write('\n')
        f.write(' '.join(str(_) for _ in model_pars['hiddenlayers'][element]) +
                '  # number of nodes of hidden-layers of %s neural network' %
                element)
        f.write('\n')

    # writing parameters of the neural network
    f.write(' '.join(str(_) for _ in \
                     # calc.model.ravel.to_vector(model_pars.weights,
                     # model_pars.scalings)
                     calc.model.vector) +
            '  # weights, biases, and scalings of neural networks')
    f.write('\n')
    f.close()
