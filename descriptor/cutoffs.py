#!/usr/bin/env python
"""
This script contains different cutoff function forms.

"""

import numpy as np


class Cosine(object):

    """Cosine functional form suggested by Behler.

    :param Rc: Radius above which neighbor interactions are ignored.
    :type Rc: float
    """

    def __init__(self, Rc):

        self.Rc = Rc

    def __call__(self, Rij):
        """
        :param Rij: Distance between pair atoms.
        :type Rij: float

        :returns: float -- the vaule of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return 0.5 * (np.cos(np.pi * Rij / self.Rc) + 1.)

    def prime(self, Rij):
        """
        Derivative of the Cosine cutoff function.

        :param Rij: Distance between pair atoms.
        :type Rij: float

        :returns: float -- the vaule of derivative of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return -0.5 * np.pi / self.Rc * np.sin(np.pi * Rij / self.Rc)


class Polynomial(object):

    """Polynomial functional form suggested by Khorshidi and Peterson.

    :param gamma: The power of polynomial.
    :type gamma: float

    :param Rc: Radius above which neighbor interactions are ignored.
    :type Rc: float
    """

    def __init__(self, gamma, Rc):
        self.gamma = gamma
        self.Rc = Rc

    def __call__(self, Rij):
        """
        :param Rij: Distance between pair atoms.
        :type Rij: float

        :returns: float -- the vaule of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            value = 1. + self.gamma * (Rij / self.Rc) ** (self.gamma + 1) - \
                (self.gamma + 1) * (Rij / self.Rc) ** self.gamma
            return value

    def prime(self, Rij):
        """
        Derivative of the Cosine cutoff function.

        :param Rc: Radius above which neighbor interactions are ignored.
        :type Rc: float
        :param Rij: Distance between pair atoms.
        :type Rij: float

        :returns: float -- the vaule of derivative of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            value = (self.gamma * (self.gamma + 1) / self.Rc) * \
                ((Rij / self.Rc) ** self.gamma -
                 (Rij / self.Rc) ** (self.gamma - 1))
            return value
