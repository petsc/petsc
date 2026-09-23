.. _petsc_python_types:

PETSc Python types
==================

PETSc supports Python implementations of matrices, preconditioners, Krylov
solvers, nonlinear solvers, ODE integrators, optimizers, and viewers.

The low-level, Cython implementation exposing the Python methods is in `src/petsc4py/PETSc/libpetsc4py.pyx <https://gitlab.com/petsc/petsc/-/tree/release/src/binding/petsc4py/src/petsc4py/PETSc/libpetsc4py.pyx>`_.

The scripts used here can be found at `demo/python_types <https://gitlab.com/petsc/petsc/-/tree/release/src/binding/petsc4py/demo/python_types>`_.

Each implementation is a Python object, called a *context*, whose methods PETSc
calls to perform the supported operations. The protocol classes below describe
the callback signatures; they are not base classes to inherit from. Implement
only the callbacks your context needs, and omit unused methods. An empty method
still counts as an implementation and can override PETSc's default behavior.

The optional ``create(obj)`` callback runs when the context is attached to a
PETSc object. The optional ``destroy(obj)`` callback releases resources before
the context is replaced or removed, including when the PETSc object is
destroyed. These callbacks are distinct from ``setUp(obj)`` and, where
supported, ``reset(obj)``, which prepare and reset an object for reuse.

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

From ``demo/python_types``, we can run the script used to test our matrix
class and use command line arguments to specify that our preconditioner
should be used:

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
    linear system matrix, which is also used to construct the preconditioner:
    Mat Object: 1 MPI process
      type: python
      rows=256, cols=256
          Python: __main__.Poisson2D

.. _petsc_python_ksp:

PETSc Python linear solver type
-------------------------------

The protocol for the `petsc4py.PETSc.KSP.Type.PYTHON` Krylov solver is:

.. literalinclude:: ../../demo/python_types/ksppython_protocol.py

The following example implements one step of preconditioned Richardson
iteration:

.. math::

  r_k = b - A x_k, \qquad z_k = P^{-1} r_k, \qquad
  x_{k+1} = x_k + \omega z_k.

The ``step()`` method updates the solution. By omitting ``solve()``, the context
uses the default KSPPYTHON loop to compute residuals, check convergence, and
call monitors. This example uses left preconditioning and the unpreconditioned
residual norm. Work vectors are allocated in ``setUp()`` and released in
``reset()`` and ``destroy()``. The relaxation factor defaults to :math:`\omega = 1`
and can be changed with ``-ksp_richardson_scale omega`` as implemented in
``setFromOptions()``.

.. literalinclude:: ../../demo/python_types/ksp.py

From ``demo/python_types``, we can run the matrix example with the Python
Richardson solver and Jacobi preconditioning. The ``-ksp_view`` option displays
the solver type, Python context, and relaxation factor reported by ``view()``:

.. code-block:: console

  $ python mat.py -ksp_type python -ksp_python_type ksp.Richardson \
      -pc_type jacobi -ksp_view
  KSP Object: 1 MPI process
    type: python
      Python: ksp.Richardson
      relaxation factor: 1
    maximum iterations=10000, initial guess is zero
    tolerances: relative=1e-05, absolute=1e-50, divergence=10000.
    left preconditioning
    using UNPRECONDITIONED norm type for convergence test
  PC Object: 1 MPI process
    type: jacobi
      type DIAGONAL
    linear system matrix, which is also used to construct the preconditioner:
    Mat Object: 1 MPI process
      type: python
      rows=256, cols=256
          Python: __main__.Poisson2D

.. _petsc_python_snes:

PETSc Python nonlinear solver type
----------------------------------

The protocol for the `petsc4py.PETSc.SNES.Type.PYTHON` nonlinear solver is:

.. literalinclude:: ../../demo/python_types/snespython_protocol.py

The following example implements the complete nonlinear solve with
``scipy.optimize.root``. It solves :math:`x^2 - 2 = 0` on one process.

.. literalinclude:: ../../demo/python_types/snes.py

.. _petsc_python_ts:

PETSc Python ODE integrator type
--------------------------------

The protocol for the `petsc4py.PETSc.TS.Type.PYTHON` ODE integrator is:

.. literalinclude:: ../../demo/python_types/tspython_protocol.py

The following example implements each time step with
``scipy.integrate.odeint``. It solves

.. math::

  \frac{du}{dt} = -u, \qquad u(0) = 1

on one process.

.. literalinclude:: ../../demo/python_types/ts.py

.. _petsc_python_tao:

PETSc Python optimization solver type
-------------------------------------

The protocol for the `petsc4py.PETSc.TAO.Type.PYTHON` TAO optimizer is:

.. literalinclude:: ../../demo/python_types/taopython_protocol.py

The following example implements a gradient descent solver to minimize
:math:`f(x) = (x_0 - 1)^2 + (x_1 - 2)^2` on one process, starting from
:math:`x = (0.5, 0.5)`.

It uses a `petsc4py.PETSc.TAOLineSearch.Type.UNIT` line search with step size
:math:`0.2`, giving the update

.. math::

  x^{k+1} = x^k - 0.2 \nabla f(x^k).

.. literalinclude:: ../../demo/python_types/tao.py

.. _petsc_python_viewer:

PETSc Python viewer
-------------------

The protocol for the `petsc4py.PETSc.Viewer.Type.PYTHON` viewer is:

.. literalinclude:: ../../demo/python_types/petscviewerpython_protocol.py
