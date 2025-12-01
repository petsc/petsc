.. _petsc_python_types:

PETSc Python types
==================

Here we discuss details about Python-aware PETSc types that can be used within the library.

In particular, we discuss matrices, preconditioners, Krylov solvers, nonlinear solvers, ODE integrators and viewers.

The low-level, Cython implementation exposing the Python methods is in `src/petsc4py/PETSc/libpetsc4py.pyx <https://gitlab.com/petsc/petsc/-/tree/release/src/binding/petsc4py/src/petsc4py/PETSc/libpetsc4py.pyx>`_.

The scripts used here can be found at `demo/python_types <https://gitlab.com/petsc/petsc/-/tree/release/src/binding/petsc4py/demo/python_types>`_.

.. _petsc_python_mat:

PETSc Python matrix type
------------------------

PETSc provides a convenient way to compute the action of linear operators coded
in Python through the `petsc4py.PETSc.Mat.Type.PYTHON` type.

In addition to the matrix action, the implementation can expose additional
methods for use within the library. A template class for
the supported methods is given below.

.. literalinclude:: ../../demo/python_types/matpython_protocol.py

In the example below, we create an operator that applies the Laplacian operator
on a two-dimensional grid, and use it to solve the associated linear system.
The default preconditioner in the script is `petsc4py.PETSc.PC.Type.JACOBI`
which needs to access the diagonal of the matrix.

.. literalinclude:: ../../demo/python_types/mat.py

.. _petsc_python_pc:

PETSc Python preconditioner type
--------------------------------

The protocol for the `petsc4py.PETSc.PC.Type.PYTHON` preconditioner is:

.. literalinclude:: ../../demo/python_types/pcpython_protocol.py

In the example below, we create a Jacobi preconditioner, which needs to access
the diagonal of the matrix. The action of the preconditioner consists of the
pointwise multiplication of the inverse diagonal with the input vector.

.. literalinclude:: ../../demo/python_types/pc.py

We can run the script used to test our matrix class and use command line
arguments to specify that our preconditioner should be used:

.. code-block:: console

  $ python mat.py -pc_type python -pc_python_type pc.myJacobi -ksp_view
  KSP Object: 1 MPI process
    type: cg
    maximum iterations=10000, initial guess is zero
    tolerances: relative=1e-05, absolute=1e-50, divergence=10000.
    left preconditioning
    using PRECONDITIONED norm type for convergence test
  PC Object: 1 MPI process
    type: python
      Python: pc.myJacobi
    linear system matrix = precond matrix:
    Mat Object: 1 MPI process
      type: python
      rows=256, cols=256
          Python: __main__.Poisson2D

.. _petsc_python_ksp:

PETSc Python linear solver type
-------------------------------

The protocol for the `petsc4py.PETSc.KSP.Type.PYTHON` Krylov solver is:

.. literalinclude:: ../../demo/python_types/ksppython_protocol.py

.. _petsc_python_snes:

PETSc Python nonlinear solver type (TODO)
-----------------------------------------

.. _petsc_python_ts:

PETSc Python ode-integrator type (TODO)
---------------------------------------

.. _petsc_python_tao:

PETSc Python optimization solver type
-------------------------------------

The protocol for the `petsc4py.PETSc.TAO.Type.PYTHON` TAO optimizer is:

.. literalinclude:: ../../demo/python_types/tao.py

In the example below, we create a simple gradient based (first order)
optimization solver. A `petsc4py.PETSc.TAOLineSearch.Type.UNIT` line search
with step size :math:`0.2` is used. Therefore the update becomes

.. math::

  x^{k+1} = x^k + 0.2 \nabla f(x^k).


.. note::

  This setup is also well suited for non-linesearch-based quasi-Newton
  optimization algorithms. It provides a general interface for using the TAO
  provided state and functionality on a custom algorithm.

The optimizer can be used from Python as

.. code-block:: python

  PETSc.TAO().createPython(myGradientDescent())

or selected through the PETSc options as 

.. code-block:: console

  python tao.py -tao_type python -tao_python_type tao.myGradientDescent

.. tip::

  The prefix **tao** to **tao_python_type** is dependant on the Python module in
  which the optimizer is located. It aligns with the fully qualified Python
  module name.

.. _petsc_python_viewer:

PETSc Python viewer
-------------------

The protocol for the `petsc4py.PETSc.Viewer.Type.PYTHON` viewer is:

.. literalinclude:: ../../demo/python_types/petscviewerpython_protocol.py
