.. _Building:

==================================
Building modules
==================================

Amp is designed to be modular, so if you think you have a great descriptor scheme or machine-learning model, you can try it out.
This page describes how to add your own modules; starting with the bare-bones requirements to make it work, and building up with how to construct it so it integrates with respect to parallelization, etc.

----------------------------------
Descriptor: minimal requirements
----------------------------------

To build your own descriptor, it needs to have certain minimum requirements met, in order to play with *Amp*. The below code illustrates these minimum requirements::

    from ase.calculators.calculator import Parameters

    class MyDescriptor(object):

        def __init__(self, parameter1, parameter2):
            self.parameters = Parameters({'mode': 'atom-centered',})
            self.parameters.parameter1 = parameter1
            self.parameters.parameter2 = parameter2
    
        def tostring(self):
            return self.parameters.tostring()

        def calculate_fingerprints(self, images, cores, log):
            # Do the calculations...
            self.fingerprints = fingerprints  # A dictionary.


The specific requirements, illustrated above, are:

* Has a parameters attribute (of type `ase.calculators.calculator.Parameters`), which holds the minimum information needed to re-build your module. 
  That is, if your descriptor has user-settable parameters such as a cutoff radius, etc., they should be stored in this dictionary.
  Additionally, it must have the keyword "mode"; which must be set to either "atom-centered" or "image-centered".
  (This keyword will be used by the model class.)

* Has a "tostring" method, which converts the minimum parameters into a dictionary that can be re-constructed using `eval`.
  If you used the ASE `Parameters` class above, this class is simple::

    def tostring():
        return self.parameters.tostring()

* Has a "calculate_fingerprints" method.
  The images argument is a dictionary of training images, with keys that are unique hashes of each image in the set produced with `amp.utilities.hash_images`.
  The log is a `amp.utilities.Logger` instance, that the method can optionally use as `log('Message.')`.
  The cores keyword describes parallelization, and can safely be ignored if serial operation is desired.
  This method must save a sub-attribute `self.fingerprints` (which will be accessible in the main *Amp* instance as `calc.descriptor.fingerprints`) that contains a dictionary-like object of the fingerprints, indexed by the same keys that were in the images dictionary. 
  Ideally, `descriptor.fingerprints` is an instance of `amp.utilities.Data`, but probably any mapping (dictionary-like) object will do.

  A fingerprint is a vector.
  In **image-centered** mode, there is one fingerprint for each image. 
  This will generally be just the Cartesian positions of all the atoms in the system, but transformations are possible.
  For example this could be accessed by the images key

  >>> calc.descriptor.fingerprints[key]
  >>> [3.223, 8.234, 0.0322, 8.33]

  In **atom-centered** mode, there is a fingerprint for each atom in the image.
  Therefore, calling `calc.descriptor.fingerprints[key]` returns a list of fingerprints, in the same order as the atom ordering in the original ASE atoms object.
  So to access an individual atom's fingerprints one could do

  >>> calc.descriptor.fingerprints[key][index]
  >>> ('Cu', [8.832, 9.22, 7.118, 0.312])

  That is, the first item is the element of the atom, and the second is a 1-dimensional array which is that atom's fingerprint.
   Thus, `calc.descriptor.fingerprints[hash]` gives a list of fingerprints, in the same order the atoms appear in the image they were fingerprinted from.

