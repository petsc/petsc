.. _petsc_options:

Working with PETSc options
==========================

A very powerful feature of PETSc is that objects can be configured via command-line options.
In this way, one can choose the method to be used or set different parameters without changing the source code.
See the PETSc `manual <the_options_database>` for additional information.

In order to use command-line options in a petsc4py program, it is important to initialize the module as follows:

.. code-block:: python

  # We first import petsc4py and sys to initialize PETSc
  import sys, petsc4py
  petsc4py.init(sys.argv)

  # Import the PETSc module
  from petsc4py import PETSc

Then one can provide command-line options when running a script:

.. code-block:: console

  $ python foo.py -ksp_type gmres -ksp_gmres_restart 100 -ksp_view

When the above initialization method is not possible, PETSc options can be also specified via environment variables or configuration files, e.g.:

.. code-block:: console

  $ PETSC_OPTIONS='-ksp_type gmres -ksp_gmres_restart 100 -ksp_view' python foo.py

Command-line options can be read via an instance of the ``Options`` class. For instance:

.. code-block:: python

  OptDB = PETSc.Options()
  n     = OptDB.getInt('n', 16)
  eta   = OptDB.getReal('eta', 0.014)
  alpha = OptDB.getScalar('alpha', -12.3)

In this way, if the script is run with

.. code-block:: console

  $ python foo.py -n 50 -alpha 8.8

the options, ``n`` and ``alpha`` will get the values ``50`` and ``8.8``, respectively, while ``eta`` will be assigned the value specified as default, ``0.014``.

The options database is accessible also as a Python dictionary, so that one can for instance override, insert or delete an option:

.. code-block:: python

  OptDB['draw_pause'] = 1
  del OptDB['draw_pause']
