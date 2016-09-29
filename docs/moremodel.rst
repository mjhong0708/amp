.. _ExampleScripts:


==================================
More on models
==================================

It can be useful to visualize the neural network model to see how it is behaving. For example, you may find nodes that are effectively shut off (e.g., always giving a constant value like 1) or that are acting as a binary switch (e.g., only returning 1 or -1). There is a tool to allow you to visualize the node outputs of a set of data.

.. code-block:: python

    from amp.model.neuralnetwork import NodePlot

    nodeplot = NodePlot(calc)
    nodeplot.plot(images, filename='nodeplottest.pdf')


This will create a plot that looks something like below. Note that one such series of plots is made for each element. Here, Layer 0 is the input layer, from the fingerprints. Layer 1 and Layer 2 are the hidden layers. Layer 3 is the output layer; that is, the contribution of Pt to the potential energy (before it is multiplied by and added to a parameter to bring it to the correct magnitude).

.. image:: _static/nodeplot-Pt.svg
   :scale: 80 %
   :align: center
