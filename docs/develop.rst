.. _Develop:

==================================
Development
==================================

This page contains standard practices for developing Amp, focusing on repositories and documentation.

----------------------------------
Repositories and branching
----------------------------------

The main Amp repository lives on bitbucket, andrewpeterson/amp.
We employ a branching model where the `master` branch is the main development branch, containing day-to-day commits from the core developers and honoring merge requests from others From time to time, we create a new branch that corresponds to a release. This release branch contains only the tagged release and any bug fixes.

   .. image:: _static/branches.svg
      :scale: 100 %
      :align: center


----------------------------------
Contributing
----------------------------------

You are welcome to contribute new features, bug fixes, better documentation, etc. to Amp. If you would like to contribute, please create a private fork and a branch for your new commits. When it is ready, send us a merge request. We follow the same basic model as ASE; please see the ASE documentation for complete instructions.

It is also a good idea to send us an email if you are planning something complicated.

----------------------------------
Documentation
----------------------------------

This documentation is built with sphinx.
(Mkdocs doesn't seem to support autodocumentation.)
To build a local copy, cd into the docs directory and try a command such as

.. code-block:: bash

   sphinx-build . /tmp/ampdocs

This uses the style "bizstyle"; if you find this is missing on your system, you can likely install it with

.. code-block:: bash

   pip install --user sphinxjp.themes.bizstyle

