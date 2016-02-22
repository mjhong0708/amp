.. _Develop:

==================================
Development
==================================

This page contains standard practices for developing Amp, focusing on repositories and documentation.

----------------------------------
Repositories and branching
----------------------------------

The main Amp repository lives on bitbucket, andrewpeterson/amp.
We employ a branching model where the `master` branch is the main development branch, containing day-to-day commits from the core developers and honoring merge requests from others. From time to time, we create a new branch that corresponds to a release. This release branch contains only the tagged release and any bug fixes.

   .. image:: _static/branches.svg
      :scale: 100 %
      :align: center


----------------------------------
Contributing
----------------------------------

You are welcome to contribute new features, bug fixes, better documentation, etc. to Amp. If you would like to contribute, please create a private fork and a branch for your new commits. When it is ready, send us a merge request. We follow the same basic model as ASE; please see the ASE documentation for complete instructions.

As good coding practice, makes sure your code passes both the pyflakes and pep8 tests. (On linux, you should be able to run `pyflakes file.py` and `pep8 file.py`.)
If adding a new feature: consider adding a (very brief) test to the tests folder to ensure your new code continues to work and write some documentation.

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


----------------------------------
Releases
----------------------------------

To create a release, we go through the following steps.

* Create a new branch on the bitbucket repository with the version name, as in `v0.5`. (Don't create a separate branch if this is a bugfix release, e.g., v0.5.1 --- just add those to the v0.5 branch.) All subsequent work is in the new branch.

* Change `docs/conf.py`'s version information to match the new version number. 

* Change the version that prints out in the Amp headers. 

* Add the version to readthedocs' available versions.

* Change the nightly tests to test this version as the stable build.

* Tag the release with the release number, e.g., v0.5 or v0.5.1, the latter being for bug fixes.

* Create a DOI for the release and a copy on Xenodo.
