
"""
Contains functions to help parallelization. Ths first functions are just a
demonstration of the approach.
"""

class Distributor:
    """Distributes code to child executables."""

    def __init__(self, nodedict):
        """Nodedict should be a dictionary form of the nodes available,
        with the hostname as the key and the number of allowed processes
        as the value; e.g., {'localhost': 4, ...}."""
        self.nodedict = nodedict

    def call(self, fxnpath, commandlist):
        """This will be significantly re-jiggerred in structure later; just
        a toy setup now to see if it will work. fxnpath is the name of the
        executable file, and commandlist is the list of commands to be
        passed to the executable."""
