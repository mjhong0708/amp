.. _Analysis:


==================================
Analysis
==================================

----------------------------------
Convergence plots
----------------------------------

You can use the tool called `amp-plotconvergence` to help you examine the output of an Amp log file. Run `amp-plotconvergence -h` for help at the command line.

You can also access this tool as :func:`~amp.analysis.plot_convergence` from the :mod:`amp.analysis` module.

   .. image:: _static/convergence.svg
      :width: 600 px
      :align: center

----------------------------------
Other plots
----------------------------------

There are several other plotting tools within the :mod:`amp.analysis` module, including :func:`~amp.analysis.plot_parity_and_error` for making parity and error plots, and :func:`~amp.analysis.plot_sensitivity` for examining the sensitivity of the model output to the model parameters.
These modules should produce plots like below; in the order parity, error, and sensitivity from left to right.
See the module autodocumentation for details.

   .. image:: _static/parity_error_sensitivity.svg
      :width: 1000 px
      :align: center

