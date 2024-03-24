.. _petsc_options:

Working with PETSc options
==========================

A very powerful feature of PETSc is that objects can be configured via command-line options. In this way, one can choose the method to be used or set different parameters.

In order to use command-line options in a petsc4py program, it is important to initialize the module as follows:

.. code-block:: python

  # We first import petsc4py and sys to initialize PETSc
  import sys, petsc4py
  petsc4py.init(sys.argv)

  # Import the PETSc module
  from petsc4py import PETSc

Then one can provide command-line options when running the script:

.. code-block:: console

  $ python ex1.py -ksp_type gmres -ksp_gmres_restart 100 -ksp_view

Note that in order to configure a given object from the command-line options, the ``setFromOptions()`` method must be called, that is:

.. code-block:: python

  ksp.setFromOptions()

It is also possible to add new, user-defined options, via the ``Options`` class. For instance:

.. code-block:: python

  OptDB = PETSc.Options()
  n     = OptDB.getInt('n', 16)
  eta   = OptDB.getReal('eta', 0.014)
  alpha = OptDB.getScalar('alpha', -12.3)

In this way, if the program is run with the following options, ``n`` and ``alpha`` will get the values ``50`` and ``8.8``, respectively, while ``eta`` will be assigned the value specified as default, ``0.014``.

.. code-block:: console

  $ python ex1.py -n 50 -alpha 8.8

The options database is accessible also as a Python dictionary, so that one can for instance override an option or insert a new option:

.. code-block:: python

  OptDB['draw_pause'] = 1
