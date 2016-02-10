.. _Develop:

==================================
Development
==================================

----------------------------------
Contributing
----------------------------------

Your bugfixes and enhancements are welcome!
Simply create a fork of our project via bitbucket, then send us a merge request.
Or, if it's something simple, feel free to just email us the changed code and we can manually include it.

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

