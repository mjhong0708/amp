.. _ExampleScripts:


Neural network model
====================

Observer
--------

When training the neural network, if you would like to model the progress you can do so with an observer. To do so, attach it before you train with something like

.. code-block:: python

  def observer(model, vector, loss):
      """This function can extract data during optimization.
      Full access is provided to the model, the vector of parameters, and
      the current value of the loss function.
      """
      pass

  calc.model.observer = myobserver


Your function "observer" will be called at each call to the loss function. For example, you can use this to print out values of specific parameter functions.
