"""
Exact Gaussian-neural scheme loss function, energy loss and force loss
for five different non-periodic configurations and three three different
periodic configurations have been calculated in Mathematica. This script
checks the values calculated by the code during training with and without
fortran modules and also on different number of cores.

"""


import numpy as np
from collections import OrderedDict
from ase import Atoms
from ase.calculators.emt import EMT
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


# The test function for non-periodic systems

convergence = {'energy_rmse': 10.**10.,
               'energy_maxresid': 10.**10.,
               'force_rmse': 10.**10.,
               'force_maxresid': 10.**10., }


def non_periodic_0th_bfgs_step_test():

    # Making the list of periodic image

    images = [Atoms(symbols='PdOPd2',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                            [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  0.,  0.],
                         [0.,  2.,  0.],
                            [0.,  0.,  3.],
                            [1.,  0.,  0.]])),
              Atoms(symbols='PdOPd2',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                            [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  1.,  0.],
                         [1.,  2.,  1.],
                            [-1.,  1.,  2.],
                            [1.,  3.,  2.]])),
              Atoms(symbols='PdO',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[2.,  1., -1.],
                         [1.,  2.,  1.]])),
              Atoms(symbols='Pd2O',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[-2., -1., -1.],
                         [1.,  2.,  1.],
                         [3.,  4.,  4.]])),
              Atoms(symbols='Cu',
                    pbc=np.array([False, False, False], dtype=bool),
                    cell=np.array(
                        [[1.,  0.,  0.],
                         [0.,  1.,  0.],
                         [0.,  0.,  1.]]),
                    positions=np.array(
                        [[0.,  0.,  0.]]))]

    for image in images:
        image.set_calculator(EMT())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)

    # Parameters

    Gs = {'O': [{'type': 'G2', 'element': 'Pd', 'eta': 0.8},
                {'type': 'G4', 'elements': [
                    'Pd', 'Pd'], 'eta':0.2, 'gamma':0.3, 'zeta':1},
                {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.3, 'gamma':0.6,
                 'zeta':0.5}],
          'Pd': [{'type': 'G2', 'element': 'Pd', 'eta': 0.2},
                 {'type': 'G4', 'elements': ['Pd', 'Pd'],
                  'eta':0.9, 'gamma':0.75, 'zeta':1.5},
                 {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.4,
                  'gamma':0.3, 'zeta':4}],
          'Cu': [{'type': 'G2', 'element': 'Cu', 'eta': 0.8},
                 {'type': 'G4', 'elements': ['Cu', 'O'],
                  'eta':0.2, 'gamma':0.3, 'zeta':1},
                 {'type': 'G4', 'elements': ['Cu', 'Cu'], 'eta':0.3,
                  'gamma':0.6, 'zeta':0.5}]}

    hiddenlayers = {'O': (2,), 'Pd': (2,), 'Cu': (2,)}

    weights = OrderedDict([('O', OrderedDict([(1, np.matrix([[-2.0, 6.0],
                                                             [3.0, -3.0],
                                                             [1.5, -0.9],
                                                             [-2.5, -1.5]])),
                                              (2, np.matrix([[5.5],
                                                             [3.6],
                                                             [1.4]]))])),
                           ('Pd', OrderedDict([(1, np.matrix([[-1.0, 3.0],
                                                              [2.0, 4.2],
                                                              [1.0, -0.7],
                                                              [-3.0, 2.0]])),
                                               (2, np.matrix([[4.0],
                                                              [0.5],
                                                              [3.0]]))])),
                           ('Cu', OrderedDict([(1, np.matrix([[0.0, 1.0],
                                                              [-1.0, -2.0],
                                                              [2.5, -1.9],
                                                              [-3.5, 0.5]])),
                                               (2, np.matrix([[0.5],
                                                              [1.6],
                                                              [-1.4]]))]))])

    scalings = OrderedDict([('O', OrderedDict([('intercept', -2.3),
                                               ('slope', 4.5)])),
                            ('Pd', OrderedDict([('intercept', 1.6),
                                                ('slope', 2.5)])),
                            ('Cu', OrderedDict([('intercept', -0.3),
                                                ('slope', -0.5)]))])

    # Correct values

    ref_loss = 7144.8107853579895
    ref_energyloss = (24.318837496016506 ** 2.) * 5
    ref_forceloss = (144.70282477494519 ** 2.) * 5
    ref_dloss_dparameters = np.array([0, 0, 0, 0, 0, 0,
                                      0.01374139170953901,
                                      0.36318423812749656,
                                      0.028312691567496464,
                                      0.6012336354445753,
                                      0.9659002689921986,
                                      -1.289777005924742,
                                      -0.5718960934643078,
                                      -2.642566722179569,
                                      -1.196039924610482, 0, 0,
                                      -2.72563797131018,
                                      -0.9080181024866707,
                                      -0.7739948323226851,
                                      -0.29157894253717415,
                                      -2.0599829042717404,
                                      -0.6156374289895887,
                                      -0.006086517460749253,
                                      -0.829678548408266,
                                      0.0008092646745710161,
                                      0.04161302703491613,
                                      0.0034264690790135606,
                                      -0.957800456897051,
                                      -0.006281929606579444,
                                      -0.2883588477371198,
                                      -4.245777410962108,
                                      -4.3174120941045535,
                                      -8.02385959091948,
                                      -3.240512651984099,
                                      -27.289862194988853,
                                      -26.8177742762544,
                                      -82.45107056051073,
                                      -80.68167683508715])
    ref_energy_maxresid = 54.21915548269209
    ref_force_maxresid = 791.6736436232306

    # Testing pure-python and fortran versions of Gaussian-neural on different
    # number of processes

    for fortran in [False, True]:
        for cores in range(1, 6):
            label = 'train-nonperiodic/%s-%i' % (fortran, cores)
            print label
            calc = Amp(descriptor=Gaussian(cutoff=6.5,
                                           Gs=Gs,
                                           fortran=fortran,),
                       model=NeuralNetwork(hiddenlayers=hiddenlayers,
                                           weights=weights,
                                           scalings=scalings,
                                           activation='sigmoid',
                                           fortran=fortran,),
                       label=label,
                       dblabel=label,
                       cores=cores)

            lossfunction = LossFunction(convergence=convergence)
            calc.model.lossfunction = lossfunction
            calc.train(images=images,)
            diff = abs(calc.model.lossfunction.loss - ref_loss)
            assert (diff < 10.**(-10.)), \
                'Calculated value of loss function is wrong!'
            diff = abs(calc.model.lossfunction.energy_loss - ref_energyloss)
            assert (diff < 10.**(-10.)), \
                'Calculated value of energy per atom RMSE is wrong!'
            diff = abs(calc.model.lossfunction.force_loss - ref_forceloss)
            assert (diff < 10 ** (-10.)), \
                'Calculated value of force RMSE is wrong!'
            diff = abs(calc.model.lossfunction.energy_maxresid -
                       ref_energy_maxresid)
            assert (diff < 10.**(-10.)), \
                'Calculated value of energy per atom max residual is wrong!'
            diff = abs(calc.model.lossfunction.force_maxresid -
                       ref_force_maxresid)
            assert (diff < 10 ** (-10.)), \
                'Calculated value of force max residual is wrong!'

            for _ in range(len(ref_dloss_dparameters)):
                diff = abs(calc.model.lossfunction.dloss_dparameters[_] -
                           ref_dloss_dparameters[_])
                assert(diff < 10 ** (-12.)), \
                    "Calculated value of loss function derivative is wrong!"

            dblabel = label
            secondlabel = '_' + label

            calc = Amp(descriptor=Gaussian(cutoff=6.5,
                                           Gs=Gs,
                                           fortran=fortran,),
                       model=NeuralNetwork(hiddenlayers=hiddenlayers,
                                           weights=weights,
                                           scalings=scalings,
                                           activation='sigmoid',
                                           fortran=fortran,),
                       label=secondlabel,
                       dblabel=dblabel,
                       cores=cores)

            lossfunction = LossFunction(convergence=convergence)
            calc.model.lossfunction = lossfunction
            calc.train(images=images,)
            diff = abs(calc.model.lossfunction.loss - ref_loss)
            assert (diff < 10.**(-10.)), \
                'Calculated value of loss function is wrong!'
            diff = abs(calc.model.lossfunction.energy_loss - ref_energyloss)
            assert (diff < 10.**(-10.)), \
                'Calculated value of energy per atom RMSE is wrong!'
            diff = abs(calc.model.lossfunction.force_loss - ref_forceloss)
            assert (diff < 10 ** (-10.)), \
                'Calculated value of force RMSE is wrong!'
            diff = abs(calc.model.lossfunction.energy_maxresid -
                       ref_energy_maxresid)
            assert (diff < 10.**(-10.)), \
                'Calculated value of energy per atom max residual is wrong!'
            diff = abs(calc.model.lossfunction.force_maxresid -
                       ref_force_maxresid)
            assert (diff < 10 ** (-10.)), \
                'Calculated value of force max residual is wrong!'

            for _ in range(len(ref_dloss_dparameters)):
                diff = abs(calc.model.lossfunction.dloss_dparameters[_] -
                           ref_dloss_dparameters[_])
                assert(diff < 10 ** (-12.)), \
                    'Calculated value of loss function derivative is wrong!'


# The test function for periodic systems and first BFGS step

def periodic_0th_bfgs_step_test():

    # Making the list of images

    images = [Atoms(symbols='PdOPd',
                    pbc=np.array([True, False, False], dtype=bool),
                    cell=np.array(
                        [[2.,  0.,  0.],
                         [0.,  2.,  0.],
                         [0.,  0.,  2.]]),
                    positions=np.array(
                        [[0.5,  1., 0.5],
                         [1.,  0.5,  1.],
                         [1.5,  1.5,  1.5]])),
              Atoms(symbols='PdO',
                    pbc=np.array([True, True, False], dtype=bool),
                    cell=np.array(
                        [[2.,  0.,  0.],
                         [0.,  2.,  0.],
                            [0.,  0.,  2.]]),
                    positions=np.array(
                        [[0.5,  1., 0.5],
                         [1.,  0.5,  1.]])),
              Atoms(symbols='Cu',
                    pbc=np.array([True, True, False], dtype=bool),
                    cell=np.array(
                        [[1.8,  0.,  0.],
                         [0.,  1.8,  0.],
                            [0.,  0.,  1.8]]),
                    positions=np.array(
                        [[0.,  0., 0.]]))]

    for image in images:
        image.set_calculator(EMT())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)

    # Parameters

    Gs = {'O': [{'type': 'G2', 'element': 'Pd', 'eta': 0.8},
                {'type': 'G4', 'elements': ['O', 'Pd'], 'eta':0.3, 'gamma':0.6,
                 'zeta':0.5}],
          'Pd': [{'type': 'G2', 'element': 'Pd', 'eta': 0.2},
                 {'type': 'G4', 'elements': ['Pd', 'Pd'],
                  'eta':0.9, 'gamma':0.75, 'zeta':1.5}],
          'Cu': [{'type': 'G2', 'element': 'Cu', 'eta': 0.8},
                 {'type': 'G4', 'elements': ['Cu', 'Cu'], 'eta':0.3,
                          'gamma':0.6, 'zeta':0.5}]}

    hiddenlayers = {'O': (2,), 'Pd': (2,), 'Cu': (2,)}

    weights = OrderedDict([('O', OrderedDict([(1, np.matrix([[-2.0, 6.0],
                                                             [3.0, -3.0],
                                                             [1.5, -0.9]])),
                                              (2, np.matrix([[5.5],
                                                             [3.6],
                                                             [1.4]]))])),
                           ('Pd', OrderedDict([(1, np.matrix([[-1.0, 3.0],
                                                              [2.0, 4.2],
                                                              [1.0, -0.7]])),
                                               (2, np.matrix([[4.0],
                                                              [0.5],
                                                              [3.0]]))])),
                           ('Cu', OrderedDict([(1, np.matrix([[0.0, 1.0],
                                                              [-1.0, -2.0],
                                                              [2.5, -1.9]])),
                                               (2, np.matrix([[0.5],
                                                              [1.6],
                                                              [-1.4]]))]))])

    scalings = OrderedDict([('O', OrderedDict([('intercept', -2.3),
                                               ('slope', 4.5)])),
                            ('Pd', OrderedDict([('intercept', 1.6),
                                                ('slope', 2.5)])),
                            ('Cu', OrderedDict([('intercept', -0.3),
                                                ('slope', -0.5)]))])

    # Correct values

    ref_loss = 8004.292841411172
    ref_energyloss = (43.7360019403031 ** 2.) * 3
    ref_forceloss = (137.40994760947325 ** 2.) * 3
    ref_dloss_dparameters = np.array([0.08141668748130322,
                                      0.03231235582925534,
                                      0.04388650395738586,
                                      0.017417514465922313,
                                      0.028431276597563077,
                                      0.011283700608814465,
                                      0.0941695726576061,
                                      -0.12322258890990219,
                                      0.12679918754154568,
                                      63.53960075374332,
                                      0.01624770019548904,
                                      -86.6263955859162,
                                      -0.01777752828707744,
                                      86.22415217526024,
                                      0.017745913074496918,
                                      104.58358033298292,
                                      -96.73280209888215,
                                      -99.09843648905876,
                                      -8.302880631972338,
                                      -1.2590007162074357,
                                      8.302877346883133,
                                      1.25875988418134,
                                      -8.302866610678247,
                                      -1.2563833805675353,
                                      28.324298392680998,
                                      28.093155094726413,
                                      -29.37874455931869,
                                      -11.247473567044866,
                                      11.119951466664787,
                                      -87.08582317481387,
                                      -20.939485239182346,
                                      -125.73267675705365,
                                      -35.138524407482116])

    # Testing pure-python and fortran versions of Gaussian-neural on different
    # number of processes

    for fortran in [False, True]:
        for cores in range(1, 4):
            label = 'train-periodic/%s-%i' % (fortran, cores)
            print label
            calc = Amp(descriptor=Gaussian(cutoff=4.,
                                           Gs=Gs,
                                           fortran=fortran,),
                       model=NeuralNetwork(hiddenlayers=hiddenlayers,
                                           weights=weights,
                                           scalings=scalings,
                                           activation='tanh',
                                           fortran=fortran,),
                       label=label,
                       dblabel=label,
                       cores=cores)

            lossfunction = LossFunction(convergence=convergence)
            calc.model.lossfunction = lossfunction
            calc.train(images=images,)
            diff = abs(calc.model.lossfunction.loss - ref_loss)
            assert (diff < 10.**(-10.)), \
                'Calculated value of loss function is wrong!'
            diff = abs(calc.model.lossfunction.energy_loss - ref_energyloss)
            assert (diff < 10.**(-10.)), \
                'Calculated value of energy per atom RMSE is wrong!'
            diff = abs(calc.model.lossfunction.force_loss - ref_forceloss)
            assert (diff < 10 ** (-9.)), \
                'Calculated value of force RMSE is wrong!'

            for _ in range(len(ref_dloss_dparameters)):
                diff = abs(calc.model.lossfunction.dloss_dparameters[_] -
                           ref_dloss_dparameters[_])
                assert(diff < 10 ** (-10.)), \
                    'Calculated value of loss function derivative is wrong!'

            dblabel = label
            secondlabel = '_' + label

            calc = Amp(descriptor=Gaussian(cutoff=4.,
                                           Gs=Gs,
                                           fortran=fortran),
                       model=NeuralNetwork(hiddenlayers=hiddenlayers,
                                           weights=weights,
                                           scalings=scalings,
                                           activation='tanh',
                                           fortran=fortran,),
                       label=secondlabel,
                       dblabel=dblabel,
                       cores=cores)

            lossfunction = LossFunction(convergence=convergence)
            calc.model.lossfunction = lossfunction
            calc.train(images=images,)
            diff = abs(calc.model.lossfunction.loss - ref_loss)
            assert (diff < 10.**(-10.)), \
                'Calculated value of loss function is wrong!'
            diff = abs(calc.model.lossfunction.energy_loss - ref_energyloss)
            assert (diff < 10.**(-10.)), \
                'Calculated value of energy per atom RMSE is wrong!'
            diff = abs(calc.model.lossfunction.force_loss - ref_forceloss)
            assert (diff < 10 ** (-9.)), \
                'Calculated value of force RMSE is wrong!'

            for _ in range(len(ref_dloss_dparameters)):
                diff = abs(calc.model.lossfunction.dloss_dparameters[_] -
                           ref_dloss_dparameters[_])
                assert(diff < 10 ** (-10.)), \
                    'Calculated value of loss function derivative is wrong!'


if __name__ == '__main__':
    non_periodic_0th_bfgs_step_test()
    periodic_0th_bfgs_step_test()
