.. _integrator_table:

==============================================
Summary of Time Integrators Available In PETSc
==============================================

.. list-table:: Time integration schemes
   :name: tab_TSPET
   :header-rows: 1

   * - TS Name
     - Reference
     - Class
     - Type
     - Order
   * - euler
     - forward Euler
     - one-step
     - explicit
     - :math:`1`
   * - ssp
     - multistage SSP :cite:`Ketcheson_2008`
     - Runge-Kutta
     - explicit
     - :math:`\le 4`
   * - rk*
     - multiscale
     - Runge-Kutta
     - explicit
     - :math:`\ge 1`
   * - beuler
     - backward Euler
     - one-step
     - implicit
     - :math:`1`
   * - cn
     - Crank-Nicolson
     - one-step
     - implicit
     - :math:`2`
   * - theta*
     - theta-method
     - one-step
     - implicit
     - :math:`\le 2`
   * - alpha
     - alpha-method :cite:`Jansen_2000`
     - one-step
     - implicit
     - :math:`2`
   * - gl
     - general linear :cite:`Butcher_2007`
     - multistep-multistage
     - implicit
     - :math:`\le 3`
   * - eimex
     - extrapolated IMEX :cite:`Constantinescu_A2010a`
     - one-step
     - IMEX
     - :math:`\ge 1`, adaptive
   * - arkimex
     - See :any:`tab_IMEX_RK_PETSc`
     - IMEX Runge-Kutta
     - IMEX
     - :math:`1-5`
   * - rosw
     - See :any:`tab_IMEX_RosW_PETSc`
     - Rosenbrock-W
     - linearly implicit
     - :math:`1-4`
   * - glee
     - See :any:`tab_IMEX_GLEE_PETSc`
     - GL with global error
     - explicit and implicit
     - :math:`1-3`
   * - mprk
     - Multirate Partitioned Runge-Kutta
     - multirate
     - explicit
     - :math:`2-3`
   * - basicsymplectic
     - Basic symplectic integrator for separable Hamiltonian
     - semi-implicit Euler and Velocity Verlet
     - explicit
     - :math:`1-2`
   * - irk
     - fully implicit Runge-Kutta
     - Gauss-Legrendre
     - implicit
     - :math:`2s`
.. bibliography:: /petsc.bib
   :filter: docname in docnames
