.. _Analysis:


==================================
Analysis
==================================

----------------------------------
Convergence plots
----------------------------------

You can use the tool called `amp-plotconvergence` to help you examine the output of an Amp log file. Run `amp-plotconvergence -h` for help at the command line.

You can also access this tool as `plot_convergence` from the `amp.analysis` module.

   .. image:: _static/convergence.svg
      :width: 600 px
      :align: center

----------------------------------
Other plots
----------------------------------

There are several other plotting tools within this module, including `plot_parity` for making parity plots, `plot_error` for making error plots, and `perturb_parameters` for examining the sensitivity of the model output to the model parameters.
These modules should produce plots like below; in the order parity, error, and sensitivity from left to right.
See the module autodocumentation for details.

   .. image:: _static/parity_error_sensitivity.svg
      :width: 600 px
      :align: center

