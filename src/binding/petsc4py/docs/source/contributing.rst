Contributing
============

Contributions from the user community are welcome. See
the `PETSc developers <ind_developers>` documentation for general
information on contributions.

New contributions to petsc4py **must** adhere with the coding standards.
We use cython-lint_ for Cython and ruff_ for Python source codes.
These can be installed using::

  $ python -m pip install -r src/binding/petsc4py/conf/requirements-lint.txt

If you are contributing Cython code, you can check compliance with::

  $ make cython-lint -C src/binding/petsc4py

For Python code, run::

  $ make ruff-lint -C src/binding/petsc4py

Python code can be auto-formatted using::

  $ make ruff-lint RUFF_OPTS='format' -C src/binding/petsc4py

New contributions to petsc4py must be tested.
Tests are located in the :file:`src/binding/petsc4py/test` folder.
To add a new test, either add a new :file:`test_xxx.py` or modify a
pre-existing file according to the
`unittest <https://docs.python.org/3/library/unittest.html>`_
specifications.

If you add a new :file:`test_xxx.py`, you can run the tests using::

  $ cd src/binding/petsc4py
  $ python test/runtests.py -k test_xxx

If instead you are modifying an existing :file:`test_xxx.py`,
you can test your additions by using the fully qualified name of the Python
class or method you are modifying, e.g.::

  $ python test/runtests.py -k test_xxx.class_name.method_name

All new code must include documentation in accordance with the `documentation
standard <documentation_standards>`. To check for compliance, run::

  $ make html SPHINXOPTS='-W' -C src/binding/petsc4py/docs/source

.. warning::

    The docstrings must not cause Sphinx warnings.

.. _cython-lint: https://github.com/MarcoGorelli/cython-lint
.. _ruff: https://docs.astral.sh/ruff
