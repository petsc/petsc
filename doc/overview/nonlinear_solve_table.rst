.. _doc_nonlinsolve:

===============================================
Summary of Nonlinear Solvers Available In PETSc
===============================================

See the paper `Composing Scalable Nonlinear Algebraic Solvers
<https://www.mcs.anl.gov/papers/P2010-0112.pdf>`__ for details
on the algorithms.

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Algorithm
     - Associated Type
     - Notes
   * - Newton's method
     - ``SNESNEWTONLS``
     - Use ``-snes_mf`` for matrix-free linear solvers
   * - Newton's method with trust region
     - ``SNESNEWTONTR``
     -
   * - Single linearization
     - ``SNESKSPONLY``
     - Essentially one step of Newtwon without a line search
   * - Quasi-Newton method (BFGS)
     - ``SNESQN``
     -
   * - Nonlinear CG
     - ``SNESNCG``
     - Requires nearly symmetric Jacobian for good convergence
   * - Nonlinear GMRES
     - ``SNESNGMRES``
     -
   * - Anderson mixing
     - ``SNESANDERSON``
     -
   * - Nonlinear Richardson
     - ``SNESNRICHARDSON``
     -
   * - Nonlinear Gauss-Siedel
     - ``SNESNGS``
     -
   * - Full Approximation Scheme (nonlinear multigrid)
     - ``SNESFAS``
     -
   * - Nonlinear additive Schwarz
     - ``SNESNASM``
     -
   * - Nonlinear additive Schwarz preconditioned inexact Newton (ASPIN) methods
     - ``SNESASPIN``
     -
   * - Composite (combine several nonlinear solvers)
     - ``SNESCOMPOSITE``
     -
   * - Preconditioned nonlinear solver
     - ---
     - See ``SNESGetNPC()``/ ``SNESSetNPC()``, can be combined to accelerate many of the solvers
